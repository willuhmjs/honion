//! The coordinator: draws every secret, trusts none of the GPUs.
//!
//! A remote worker connection is handled exactly the way
//! [`crate::search::search_device`] handles a local device — draw epoch
//! scalars here, ship only their public keys, re-derive and re-verify every
//! claim here, write every key here — with the device work itself moved one
//! socket away. The coordinator's secrets never approach that socket:
//! [`Msg::Epoch`] carries compressed curve points, and [`Msg::Report`]
//! carries `(thread, offset)` claims that are only ever inputs to the same
//! host-side verifier a flaky local GPU's hits already pass through.
//!
//! # Failure accounting, deliberately asymmetric
//!
//! A local worker that dies stops the run, because its failure means the
//! machine's search is damaged. A *remote* worker that hangs up is the
//! ordinary case of that worker's operator walking away: if the run was
//! already stopping (or stopped while the socket closed) the disconnect is
//! silence; otherwise it is reported and the remaining workers keep
//! searching. Nobody rented the coordinator a GPU that cannot leave.
//!
//! # Threading
//!
//! The connection threads are ordinary `thread::spawn`s over an
//! [`Arc`][std::sync::Arc]-shared [`CoordCtx`] rather than scoped threads:
//! scoped threads cannot be spawned from inside a scoped thread (the
//! borrow checker correctly refuses the lifetime game), and the `Arc` shape
//! is what an unbounded-lifetime connection thread needs anyway. The
//! overseer's [`std::thread::JoinHandle`] is joined by `run_search` before
//! it reports, so "the search is over" still means every in-flight claim
//! has been verified or dropped, never abandoned mid-write.

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use honion_core::pattern::{PatternSet, SuffixSet};
use honion_keyfile::{StoreError, VerifiedKey, write_service_dir};

use super::proto::{self, Msg};
use crate::search::{self, SearchArgs, Shared};
use crate::state::SearchState;

/// The largest thread count a worker may promise.
///
/// The bound is the wire: an epoch carries 32 bytes per thread inside a
/// [`proto::MAX_PAYLOAD`]-capped frame, and promising more would make the
/// coordinator build an epoch it cannot send. Two million threads is already
/// far past any single device — the CUDA `auto_threads` cap is a quarter of
/// it — so this refuses only nonsense, never hardware.
const MAX_WORKER_THREADS: u32 = 2_000_000;

/// Everything a connection thread needs, cheap to hand to an unscoped
/// thread and to keep alive exactly as long as that thread does.
pub(crate) struct CoordCtx {
    pub(crate) args: Arc<SearchArgs>,
    pub(crate) set: Arc<PatternSet>,
    pub(crate) suffixes: Arc<SuffixSet>,
    pub(crate) meta: Arc<SearchState>,
    pub(crate) shared: Arc<Shared>,
}

/// Spawn the coordinator's overseer: accept connections, give each a
/// thread, and on shutdown join every one of them.
///
/// The returned handle is what the caller joins to know that no remote
/// round is still in flight; the overseer itself exits when the shared
/// `stop` flag is set.
pub(crate) fn serve(ctx: CoordCtx, listener: TcpListener) -> Result<std::thread::JoinHandle<()>> {
    listener
        .set_nonblocking(true)
        .context("setting the coordinator's listener non-blocking")?;
    std::thread::Builder::new()
        .name("honion-coord".into())
        .spawn(move || overseer(ctx, listener))
        .context("spawning the coordinator accept loop")
}

/// Accept until `stop`, then reap.
fn overseer(ctx: CoordCtx, listener: TcpListener) {
    let mut workers: Vec<(std::net::SocketAddr, std::thread::JoinHandle<()>)> = Vec::new();
    loop {
        if ctx.shared.stop.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, peer)) => {
                let ctx = CoordCtx {
                    args: Arc::clone(&ctx.args),
                    set: Arc::clone(&ctx.set),
                    suffixes: Arc::clone(&ctx.suffixes),
                    meta: Arc::clone(&ctx.meta),
                    shared: Arc::clone(&ctx.shared),
                };
                let spawn = std::thread::Builder::new()
                    .name(format!("honion-worker@{peer}"))
                    .spawn(move || {
                        if let Err(e) = handle_worker(stream, peer, &ctx) {
                            // A worker walking away mid-run is reported —
                            // but see the module docs: it does not stop the
                            // rest.
                            if !ctx.shared.stop.load(Ordering::Relaxed) {
                                eprintln!("error: worker {peer}: {e:?}");
                                ctx.shared
                                    .errors
                                    .lock()
                                    .expect("the error list was poisoned")
                                    .push(format!("worker {peer}"));
                            }
                        }
                    });
                match spawn {
                    Ok(handle) => workers.push((peer, handle)),
                    Err(e) => eprintln!("error: accepting worker {peer}: {e}"),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("error: coordinator accept: {e}");
                break;
            }
        }
    }
    for (peer, handle) in workers {
        // Each connection thread bounds its own waits — the run is stopping
        // and must not hang on a worker that has stopped answering.
        if handle.join().is_err() {
            eprintln!("error: worker {peer} thread panicked");
        }
    }
}

/// One worker connection, start to finish.
fn handle_worker(mut stream: TcpStream, peer: std::net::SocketAddr, ctx: &CoordCtx) -> Result<()> {
    let args = &ctx.args;
    stream
        .set_nodelay(true)
        .context("disabling Nagle on the worker link")?;
    // Every later read is bounded: a worker whose machine dies must free
    // this thread (and thus the overseer's join) rather than hang the
    // search's teardown forever. Generous enough for a launch interval of
    // seconds plus one slow epoch, small enough that a dead peer is
    // noticed inside a minute of a default-configured run.
    let silence = Duration::from_secs_f64((args.launch_seconds * 8.0).clamp(30.0, 300.0));
    stream
        .set_read_timeout(Some(silence))
        .context("bounding worker read waits")?;
    stream
        .set_write_timeout(Some(silence))
        .context("bounding worker write waits")?;

    let Msg::Hello { device } = proto::read_msg(&mut stream).context("awaiting hello")? else {
        bail!("worker sent something other than Hello");
    };
    if ctx.shared.stop.load(Ordering::Relaxed) {
        let _ = proto::write_msg(
            &mut stream,
            &Msg::Error("the search is already over".into()),
        );
        return Ok(());
    }

    // The worker compiles the search itself from recognised text; what
    // travels is the configuration, which the coordinator would otherwise
    // have typed at its own keyboard, and nothing it keeps secret.
    proto::write_msg(
        &mut stream,
        &Msg::Config {
            patterns: ctx.meta.prefixes.clone(),
            suffixes: ctx.meta.suffixes.clone(),
            half: args.offsets,
            launch_seconds: args.launch_seconds,
            min_selectivity: args.min_selectivity,
        },
    )
    .context("sending configuration")?;

    let Msg::Ready { threads } = proto::read_msg(&mut stream).context("awaiting ready")? else {
        bail!("worker sent something other than Ready");
    };
    if threads == 0 || threads > MAX_WORKER_THREADS {
        let why = format!("{threads} threads is outside 1..={MAX_WORKER_THREADS}");
        let _ = proto::write_msg(&mut stream, &Msg::Error(why.clone()));
        bail!(why);
    }
    if !args.quiet {
        eprintln!("  worker {peer} joined: {device}, {threads} threads");
    }

    let mut epoch_id: u64 = 0;
    let result = loop {
        if ctx.shared.stop.load(Ordering::Relaxed) {
            break Ok(());
        }
        match epoch_round(&mut stream, epoch_id, threads, ctx) {
            Ok(more) => {
                if !more {
                    break Ok(());
                }
            }
            Err(e) => break Err(e),
        }
        epoch_id += 1;
    };

    match result {
        Ok(()) => {
            let _ = proto::write_msg(&mut stream, &Msg::Done);
            if !args.quiet {
                eprintln!("  worker {peer} finished");
            }
            Ok(())
        }
        Err(e) => {
            // Tell the worker why the ground moved, then let the caller
            // decide whether anyone else needs to hear about it.
            let _ = proto::write_msg(&mut stream, &Msg::Error(format!("{e:#}")));
            // A disconnect raced with a stop is the ordinary walking-away
            // case; say nothing further.
            if ctx.shared.stop.load(Ordering::Relaxed) {
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

/// One epoch: draw, ship, await the report, verify, checkpoint.
///
/// Returns `false` when `--count` was met and the run is over.
///
/// The next epoch's scalars are drawn after the current report lands —
/// unlike the local loop, which draws while its own GPU works, because the
/// remote GPU is already working and cannot overlap the coordinator's CPU
/// work without buffering a second epoch per connection. The draw is tens
/// of milliseconds against a launch interval of seconds; deepening the
/// pipeline is a later optimisation with a measured case behind it.
fn epoch_round(stream: &mut TcpStream, id: u64, threads: u32, ctx: &CoordCtx) -> Result<bool> {
    let args = &ctx.args;
    let scalars = search::draw_scalars(threads)
        .context("drawing the epoch's secret scalars for a remote worker")?;
    proto::write_msg(
        stream,
        &Msg::Epoch {
            id,
            points: search::public_keys(&scalars),
        },
    )
    .context("shipping start points")?;

    let Msg::Report {
        id: reported,
        examined,
        claims,
    } = proto::read_msg(stream).context("awaiting the epoch report")?
    else {
        bail!("worker sent something other than a Report");
    };
    if reported != id {
        bail!("worker reported epoch {reported} while answering epoch {id}");
    }

    // Every claim is re-derived from the coordinator's own scalars before
    // it can touch the filesystem. A claim that fails verification is a
    // broken or dishonest worker's noise — dropped and logged, because
    // unlike a local device's disagreement (a bug, a fault: fatal) a remote
    // card is neither ours to trust nor ours to debug.
    let mut found = 0usize;
    let mut write_err: Option<anyhow::Error> = None;
    for &(thread, offset) in &claims {
        let Some(start) = scalars.get(thread as usize) else {
            eprintln!(
                "warning: worker claimed thread {thread} but drew epochs for {threads} threads; claim dropped"
            );
            continue;
        };
        let Ok(verified) = VerifiedKey::verify(start, i64::from(offset), &ctx.set) else {
            eprintln!(
                "warning: worker claim (thread {thread}, offset {offset}) failed verification; \
                 dropped"
            );
            continue;
        };
        if !ctx.suffixes.matches_any(verified.address().body()) {
            continue;
        }
        found += 1;
        let dir = match write_service_dir(&args.out, &verified) {
            Ok(dir) => dir,
            Err(StoreError::AlreadyExists { .. }) => {
                if !args.quiet {
                    eprintln!(
                        "  {} already exists; skipping",
                        verified.address().to_hostname()
                    );
                }
                continue;
            }
            Err(e) => {
                write_err =
                    Some(anyhow::Error::new(e).context("writing the hidden service directory"));
                break;
            }
        };
        let total = ctx.shared.written.fetch_add(1, Ordering::SeqCst) + 1;
        println!("{}", verified.address().to_hostname());
        if !args.quiet {
            eprintln!("  written to {}", dir.display());
        }
        if args.count != 0 && total >= args.count {
            record(stream, examined, found, ctx);
            return Ok(false);
        }
    }
    if let Some(e) = write_err {
        return Err(e);
    }

    record(stream, examined, found, ctx);
    Ok(true)
}

/// Fold one report into the shared progress, then checkpoint — the same
/// pair of bookkeeping steps the local loop performs at every launch
/// boundary, and the reason a coordinator's `--resume` sees remote work
/// too. Sends `Done` and sets `stop` exactly once `--count` is met.
fn record(stream: &mut TcpStream, examined: u64, found: usize, ctx: &CoordCtx) {
    let stopping = {
        let mut progress = ctx
            .shared
            .progress
            .lock()
            .expect("the progress counter was poisoned");
        progress.record(examined, found);
        progress.report();
        let total = ctx.shared.written.load(Ordering::SeqCst);
        let stopping = ctx.args.count != 0 && total >= ctx.args.count;
        if stopping {
            ctx.shared.stop.store(true, Ordering::Relaxed);
        }
        search::save_checkpoint(&ctx.args, &ctx.meta, &progress);
        stopping
    };
    if stopping {
        let _ = proto::write_msg(stream, &Msg::Done);
    }
}
