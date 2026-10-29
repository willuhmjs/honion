//! The worker: a rented GPU taking orders over TCP.
//!
//! A worker holds no secrets and writes no key files. It receives pattern
//! text (configuration, not a secret) and per-epoch lists of *public* start
//! points, runs exactly the launches a local device worker runs, and returns
//! `(thread, offset)` claims the coordinator verifies itself. Every property
//! `docs/05-security-model.md` demands of a GPU holds a fortiori for a
//! network peer several hops away.
//!
//! Launch sizing is the worker's own business: the coordinator states how
//! often it wants a report (`launch_seconds`) and the worker reproduces the
//! local loop's self-tuning — one deliberately small first launch to measure
//! the card, every later launch sized from the rate just measured.
//!
//! # Resume, and why the worker has none
//!
//! The checkpoint is the coordinator's — it owns the counters and the keys.
//! A worker that dies simply reconnects and takes orders again; the
//! coordinator's epoch loop notices the broken socket, reports the worker as
//! failed, and carries on with whatever workers remain.

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use clap::Args;
use honion_core::pattern::{PatternSet, SuffixPattern, SuffixSet};
use honion_gpu::{DeviceTables, Searcher};

use super::proto::{self, Msg};
use crate::patterns;
use crate::progress::{humantime, si};

/// Arguments to `honion worker`.
#[derive(Args, Debug)]
pub struct WorkerArgs {
    /// Coordinator address, `host:port` (a bare host gets the default port).
    ///
    /// No secret crosses this connection in either direction; the `net`
    /// module docs state exactly what does.
    #[arg(long, value_name = "ADDR")]
    pub coordinator: String,

    /// This machine's device to offer.
    #[arg(long, default_value_t = 0)]
    pub device: u32,

    /// Override the automatic thread count. Advanced: the default sizes the
    /// search to this device's free memory exactly as a local run would.
    #[arg(long)]
    pub threads: Option<u32>,

    /// Suppress the rate line.
    #[arg(long, short)]
    pub quiet: bool,
}

/// `honion worker`: connect, take orders, search, report, until `Done`.
///
/// # Errors
///
/// On connection failure, a protocol violation by the coordinator, or device
/// failure. Where the socket still works, the reason is sent to the
/// coordinator as an [`Msg::Error`] before the error is returned.
pub fn run_worker(args: &WorkerArgs) -> Result<()> {
    let addr = resolve(&args.coordinator)?;
    let socket =
        TcpStream::connect(addr).with_context(|| format!("connecting to coordinator {addr}"))?;
    socket
        .set_nodelay(true)
        .context("disabling Nagle on the coordinator link")?;
    let mut wire = socket;

    // Hello first. The name is for the coordinator's log, and nothing on
    // either end trusts it for anything else.
    let name = honion_gpu::describe_device(args.device)
        .map(|info| info.description())
        .unwrap_or_else(|_| format!("device {}", args.device));
    proto::write_msg(
        &mut wire,
        &Msg::Hello {
            device: name.clone(),
        },
    )
    .context("sending hello")?;

    let Msg::Config {
        patterns: pattern_srcs,
        suffixes: suffix_srcs,
        half,
        launch_seconds,
        min_selectivity,
    } = proto::read_msg(&mut wire).context("awaiting configuration")?
    else {
        bail!("the coordinator sent something other than Config");
    };

    // Build the same tables the coordinator builds for its own cards, from
    // the same recognised text: a second implementation of "what this
    // pattern means" must not exist, so the worker recognises exactly what
    // a local run would refuse.
    let sourced = patterns::from_args(&pattern_srcs).context("the coordinator's patterns")?;
    let patterns = sourced
        .iter()
        .map(|s| s.pattern.clone())
        .collect::<Vec<_>>();
    let set = PatternSet::compile(&patterns)
        .map_err(|e| anyhow!("compiling the coordinator's patterns: {e}"))?;
    let suffixes = {
        let mut parsed = Vec::with_capacity(suffix_srcs.len());
        for src in &suffix_srcs {
            parsed.push(
                SuffixPattern::parse(src)
                    .with_context(|| format!("the coordinator's --suffix {src:?}"))?,
            );
        }
        SuffixSet::new(parsed)
    };
    let tables = DeviceTables::build(&set);
    tables
        .check_selectivity(min_selectivity)
        .context("the coordinator's pattern set cannot be searched efficiently")?;

    let threads = match args.threads {
        Some(0) => bail!("--threads must be positive"),
        Some(n) => n,
        None => honion_gpu::auto_threads(args.device, half)
            .with_context(|| format!("sizing the search to device {}", args.device))?,
    };
    let mut searcher = Searcher::new(args.device, &tables, threads, half, 4096)
        .context("preparing the local GPU")?;

    proto::write_msg(&mut wire, &Msg::Ready { threads }).context("sending ready")?;
    if !args.quiet {
        eprintln!(
            "honion: worker ready: {name}, {threads} threads, {} pattern(s), {:.1} bits",
            sourced.len(),
            set.difficulty_log2() + suffixes.difficulty_log2(),
        );
    }

    // The local loop's pacing, unchanged: start small so the first launch
    // cannot run long on a slow card, then size every launch from the rate
    // the previous one measured.
    let mut candidates: u32 = 4096;
    let mut total_examined: u64 = 0;
    let mut epoch_count: u64 = 0;
    let started = Instant::now();

    loop {
        let msg = proto::read_msg(&mut wire).context("awaiting an epoch")?;
        let (id, points) = match msg {
            Msg::Epoch { id, points } => (id, points),
            Msg::Done => break,
            Msg::Error(reason) => bail!("the coordinator ended the run: {reason}"),
            other => bail!("the coordinator sent {other:?} where an Epoch was due"),
        };
        if points.len() != threads as usize {
            let why = format!(
                "epoch {id} carries {} start points but the worker promised {threads} threads",
                points.len()
            );
            let _ = proto::write_msg(&mut wire, &Msg::Error(why.clone()));
            bail!(why);
        }
        searcher
            .set_start_points(&points)
            .context("uploading start points")?;

        let launch_start = Instant::now();
        let outcome = searcher
            .launch(candidates)
            .context("running the search kernel")?;
        let secs = launch_start.elapsed().as_secs_f64();

        let claims: Vec<(u32, i32)> = outcome
            .hits
            .iter()
            .map(|h| (h.thread_id, h.offset))
            .collect();
        proto::write_msg(
            &mut wire,
            &Msg::Report {
                id,
                examined: outcome.examined,
                claims,
            },
        )
        .context("reporting the epoch")?;

        if outcome.total_found as usize > outcome.hits.len() {
            eprintln!(
                "warning: epoch {id} found {} candidates but only {} fit the buffer; \
                 the worker dropped the rest. The coordinator cannot see them.",
                outcome.total_found,
                outcome.hits.len(),
            );
        }

        total_examined += outcome.examined;
        epoch_count += 1;
        if !args.quiet {
            let rate = outcome.examined as f64 / secs.max(1e-9);
            eprintln!(
                "  worker: {:>9} at {:>7}/s  elapsed {:>9}  epoch {id} ({} claim(s))",
                si(total_examined as f64),
                si(rate),
                humantime(started.elapsed().as_secs_f64()),
                outcome.hits.len(),
            );
        }

        if secs > 0.0 {
            let per_candidate = secs / f64::from(candidates);
            let want = (launch_seconds / per_candidate).round();
            candidates = want.clamp(1.0, f64::from(u32::MAX / 2)) as u32;
        }
    }

    if !args.quiet {
        eprintln!(
            "honion: worker done: {epoch_count} epoch(s), {} examined",
            si(total_examined as f64),
        );
    }
    Ok(())
}

/// Resolve `host`, `host:port`, or `:port`; bare hosts and a bare port get
/// [`super::DEFAULT_PORT`] so both sides' defaults agree.
fn resolve(spec: &str) -> Result<std::net::SocketAddr> {
    let with_port = if let Some(port) = spec.strip_prefix(':') {
        format!("127.0.0.1:{port}")
    } else if spec.contains(':') {
        spec.to_owned()
    } else {
        format!("{spec}:{}", super::DEFAULT_PORT)
    };
    with_port
        .to_socket_addrs()
        .with_context(|| format!("resolving coordinator {spec:?}"))?
        .next()
        .ok_or_else(|| anyhow!("coordinator {spec:?} resolved to no address"))
}
