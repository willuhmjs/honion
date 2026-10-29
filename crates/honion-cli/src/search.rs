//! The `search`, `estimate` and `verify` subcommands.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use honion_core::address::OnionAddress;
use honion_core::pattern::{Pattern, PatternSet, SuffixPattern, SuffixSet};
use honion_gpu::{DeviceTables, Searcher};
use honion_keyfile::{SecretScalar, StoreError, VerifiedKey, write_service_dir};
use rayon::prelude::*;

use crate::patterns::{self, SourcedPattern};
use crate::progress::{Progress, humantime, si};
use crate::secure_mem;
use crate::state::SearchState;

/// Default number of precomputed offsets.
///
/// Each offset yields two candidates via the dual addition law, so a batch
/// covers `2 * 512 + 1 = 1025` candidates for one modular inversion.
/// See docs/06-performance.md for the tuning curve.
const DEFAULT_OFFSETS: u32 = 512;

/// Minimum prefilter selectivity, in bits.
///
/// Below this the device forwards candidates faster than the host can verify
/// them. Twenty bits leaves roughly a thousand per second at full throughput.
const DEFAULT_MIN_SELECTIVITY: f64 = 20.0;

/// How long each launch should take. Progress is reported between launches, so
/// this is also the update interval and the worst-case delay before a found key
/// is written.
const DEFAULT_LAUNCH_SECONDS: f64 = 4.0;

/// Stack size for per-device search workers. Each worker compiles the kernel
/// with NVRTC and lets the driver JIT the PTX on its own thread; that code
/// recurses deeply enough to overflow the 2 MiB a spawned thread gets by
/// default. Measured crashing at 2 MiB on CUDA 12 NVRTC, comfortable at this.
const WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Arguments to `honion search`.
#[derive(Args, Debug, Clone)]
pub struct SearchArgs {
    /// Pattern to search for. May be repeated.
    ///
    /// A pattern is base32 characters (a-z, 2-7), `?` for any character, and
    /// `[abc]` or `[^abc]` for a choice. It matches the start of the address.
    #[arg(long = "prefix", value_name = "PATTERN")]
    pub(crate) prefixes: Vec<String>,

    /// Ending the address must have. May be repeated; a candidate then needs
    /// the prefix *and* any one of these endings.
    ///
    /// Suffix characters at the very end of the address are part of the
    /// SHA3 checksum or the fixed version byte, which the search kernel does
    /// not compute; they are checked on the host while verifying each
    /// prefix hit. They cost no search throughput, and the difficulty model
    /// prices them accordingly: every address ends in 'd', the character
    /// before it is one of a, i, q, y, and only the rest cost 5 bits each.
    #[arg(long = "suffix", value_name = "PATTERN")]
    pub(crate) suffixes: Vec<String>,

    /// File of patterns, one per line; `#` comments and blank lines ignored.
    #[arg(long, value_name = "FILE")]
    patterns_file: Option<PathBuf>,

    /// Directory to write results into. One subdirectory per address.
    #[arg(long, short, value_name = "DIR")]
    pub(crate) out: PathBuf,

    /// CUDA devices to search on: `all`, or comma-separated device numbers.
    ///
    /// Each device runs its own search with its own fresh scalars, so the
    /// rates add. `--threads` is per device, not a total. Host preparation
    /// (drawing and deriving each epoch's starting keys) also runs per
    /// device, so a fully populated box wants CPU cores to match. Listing
    /// one device twice is allowed and time-slices two workers on it,
    /// which is slower than one worker and mostly a way to exercise the
    /// multi-worker path on a single-card machine.
    #[arg(long, default_value = "0", value_name = "DEV")]
    pub(crate) device: String,

    /// Stop after this many matches, across all devices. Zero means run until
    /// interrupted.
    #[arg(long, default_value_t = 1)]
    pub(crate) count: usize,

    /// Concurrent walks, per device. Defaults to as many as fit comfortably
    /// in that device's free memory, which is where throughput plateaus.
    #[arg(long)]
    pub(crate) threads: Option<u32>,

    /// Precomputed offsets. Each yields two candidates, so a batch covers
    /// `2N+1` keys per modular inversion. Larger uses more memory.
    #[arg(long, default_value_t = DEFAULT_OFFSETS)]
    pub(crate) offsets: u32,

    /// Seconds of GPU work per launch, which is also the progress interval.
    #[arg(long, default_value_t = DEFAULT_LAUNCH_SECONDS)]
    pub(crate) launch_seconds: f64,

    /// Minimum prefilter selectivity in bits.
    #[arg(long, default_value_t = DEFAULT_MIN_SELECTIVITY)]
    pub(crate) min_selectivity: f64,

    /// Suppress progress output.
    #[arg(long, short)]
    pub(crate) quiet: bool,

    /// Continue the checkpoint in --out: its patterns, elapsed time and
    /// counters carry over, so a crash or Ctrl-C costs at most one launch's
    /// statistics, never the run. Patterns may be omitted entirely.
    #[arg(long)]
    resume: bool,

    /// Start a fresh run in --out even though it holds a checkpoint (the
    /// checkpoint is left untouched until the first progress update).
    #[arg(long)]
    fresh: bool,

    /// Clear PR_SET_DUMPABLE for the whole process: no core dumps, and no
    /// same-uid ptrace either. The epoch scalars are already kept out of
    /// core dumps via MADV_DONTDUMP; this widens it to everything, at the
    /// cost of breaking cuda-gdb and Nsight attach. For runs measured in
    /// days, the trade is usually worth it.
    #[arg(long)]
    paranoid: bool,

    /// Accept remote workers at this address: `PORT`, `HOST:PORT` or
    /// `:PORT` (which binds every interface).
    ///
    /// Remote workers rent out a GPU and nothing else: the coordinator here
    /// keeps every secret scalar, ships only the epoch's public start
    /// points, and verifies every returned claim before a key is written,
    /// exactly as it does for a local device. The traffic is not encrypted
    /// or authenticated — it carries no secrets, but patterns are visible
    /// on the wire and a man in the middle can waste time, so the boundary
    /// is "a network you trust not to waste your time". Combine with
    /// `--device none` to run a coordinator on a machine without a GPU.
    #[arg(long, value_name = "ADDR")]
    pub(crate) listen: Option<String>,
}

/// Arguments to `honion estimate`.
#[derive(Args, Debug)]
pub struct EstimateArgs {
    /// Pattern to estimate. May be repeated.
    #[arg(long = "prefix", value_name = "PATTERN")]
    prefixes: Vec<String>,

    /// Ending to estimate. May be repeated.
    #[arg(long = "suffix", value_name = "PATTERN")]
    suffixes: Vec<String>,

    /// File of patterns, one per line.
    #[arg(long, value_name = "FILE")]
    patterns_file: Option<PathBuf>,

    /// Assumed search rate in addresses per second.
    ///
    /// Defaults to the rate measured on an RTX PRO 6000 Blackwell over 300 runs;
    /// see
    /// docs/07-benchmarks.md. Pass your own to estimate for a different card.
    #[arg(long, default_value_t = 1.2514e10)]
    rate: f64,
}

/// Gather patterns from both sources, failing on the first malformed one.
fn collect(prefixes: &[String], file: Option<&PathBuf>) -> Result<Vec<SourcedPattern>> {
    let mut all = patterns::from_args(prefixes)?;
    if let Some(path) = file {
        all.extend(patterns::from_file(path)?);
    }
    if all.is_empty() {
        bail!(
            "no patterns given; use --prefix or --patterns-file (a --suffix alone \
             cannot be searched: only prefixes reach the device prefilter)",
        );
    }
    Ok(all)
}

/// Resolve `--device` to the list of ordinals to search on.
///
/// Accepts `all` or a comma-separated list of numbers. Numbers are checked
/// against `available` now rather than failing later inside `Searcher::new`,
/// so `--device 3` on a two-card box says so before anything compiles.
/// Duplicates are allowed: two workers on one card are slow, but they are a
/// legitimate configuration and a useful way to exercise the multi-worker
/// path on single-card hardware.
fn parse_devices(spec: &str, available: u32) -> Result<Vec<u32>> {
    let devices = if spec.trim().eq_ignore_ascii_case("all") {
        (0..available).collect()
    } else {
        spec.split(',')
            .map(|part| -> Result<u32> {
                let entry = part.trim();
                if entry.is_empty() {
                    bail!("--device: {spec:?} has an empty entry");
                }
                let ordinal: u32 = entry
                    .parse()
                    .with_context(|| format!("--device: {entry:?} is not a device number"))?;
                if ordinal >= available {
                    bail!(
                        "--device: no device {ordinal}; the driver sees {available} (0..{})",
                        available.saturating_sub(1),
                    );
                }
                Ok(ordinal)
            })
            .collect::<Result<Vec<u32>>>()?
    };
    if devices.is_empty() {
        bail!("--device {spec:?} resolved to no devices; is a CUDA device visible?");
    }
    Ok(devices)
}

/// What the per-device search workers share.
pub(crate) struct Shared {
    /// Combined progress across every device, so the rate line a user reads
    /// is the machine's throughput rather than one card's.
    pub(crate) progress: Mutex<Progress>,
    /// Keys written so far, across all workers. This is what `--count`
    /// counts.
    pub(crate) written: AtomicUsize,
    /// Set when `--count` is met or a worker fails; workers check it at
    /// launch boundaries, so a stop takes effect within one launch interval
    /// and in-flight work is kept rather than discarded.
    pub(crate) stop: AtomicBool,
    /// One entry per failed worker, so every device's failure is reported
    /// rather than only the first.
    pub(crate) errors: Mutex<Vec<String>>,
}

/// `honion search`.
///
/// # Errors
///
/// If patterns are malformed, the prefilter is too weak, no GPU is available,
/// or a result cannot be written.
pub fn run_search(args: &SearchArgs) -> Result<()> {
    if args.paranoid {
        secure_mem::forbid_core_dump();
    }
    if args.resume && args.fresh {
        bail!("--resume and --fresh contradict each other");
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output directory {}", args.out.display()))?;

    // Checkpoint resolution, before any parsing work is duplicated: the
    // state file is *input like any other* -- its pattern strings go back
    // through the grammar, and if the command line also names patterns they
    // must agree, because silently searching different patterns than the
    // checkpoint describes would corrupt the very counters --resume exists
    // to preserve.
    let prior = SearchState::load(&args.out)?;
    if args.resume && prior.is_none() {
        bail!(
            "--resume: {} holds no checkpoint (looked for {})",
            args.out.display(),
            SearchState::path(&args.out).display()
        );
    }
    if prior.is_some() && !args.resume && !args.fresh {
        bail!(
            "{} holds a checkpoint from an earlier run; use --resume to continue \
             it or --fresh to start a new search here",
            SearchState::path(&args.out).display()
        );
    }
    let user_named_patterns =
        !args.prefixes.is_empty() || args.patterns_file.is_some() || !args.suffixes.is_empty();
    let (prefix_srcs, suffix_srcs) = if let Some(st) = &prior {
        if args.resume && !user_named_patterns {
            // The whole point: resume without retyping anything.
            (st.prefixes.clone(), st.suffixes.clone())
        } else if args.resume {
            let given_p = collect(&args.prefixes, args.patterns_file.as_ref())?;
            let given_p: Vec<&str> = given_p.iter().map(|s| s.pattern.source()).collect();
            let same_patterns =
                given_p == st.prefixes.iter().map(String::as_str).collect::<Vec<_>>();
            if !(same_patterns && args.suffixes == st.suffixes) {
                bail!(
                    "the command line's patterns differ from the checkpoint in {}; \
                     --resume continues those patterns, a new search belongs in a \
                     new --out (or --fresh to overwrite the checkpoint here)",
                    args.out.display()
                );
            }
            (
                given_p.iter().map(|s| (*s).to_owned()).collect(),
                args.suffixes.clone(),
            )
        } else {
            // --fresh: the checkpoint exists but the user explicitly chose
            // to start over, so it is history, not input -- not compared.
            (
                collect(&args.prefixes, args.patterns_file.as_ref())?
                    .iter()
                    .map(|s| s.pattern.source().to_owned())
                    .collect(),
                args.suffixes.clone(),
            )
        }
    } else if user_named_patterns {
        (
            collect(&args.prefixes, args.patterns_file.as_ref())?
                .iter()
                .map(|s| s.pattern.source().to_owned())
                .collect(),
            args.suffixes.clone(),
        )
    } else {
        // No checkpoint, no patterns: the original error, original place.
        collect(&args.prefixes, args.patterns_file.as_ref())?;
        unreachable!("collect always bails on empty input")
    };
    let sourced = patterns::from_args(&prefix_srcs).context("parsing patterns")?;
    if let Some(st) = &prior {
        if args.resume && args.count != 0 && st.written as usize >= args.count {
            bail!(
                "the checkpoint already holds {} key(s) and --count is {};                  raise --count or start in a new --out",
                st.written,
                args.count
            );
        }
    }
    let set = compile(&sourced)?;
    let tables = DeviceTables::build(&set);
    let suffixes = parse_suffixes(&suffix_srcs)?;

    // Refuse a hopeless configuration before allocating anything.
    tables
        .check_selectivity(args.min_selectivity)
        .context("this pattern set cannot be searched efficiently")?;

    if args.offsets == 0 {
        bail!("--offsets must be positive");
    }
    // `--device none` is the coordinator with no GPU of its own: a rented
    // fleet alone. It requires --listen, and it deliberately never asks a
    // CUDA driver what it thinks, so the coordinator runs on machines that
    // have no GPU answer to give.
    let devices = if args.device.trim().eq_ignore_ascii_case("none") {
        if args.listen.is_none() {
            bail!("--device none means remote workers only; add --listen to accept them");
        }
        Vec::new()
    } else {
        parse_devices(
            &args.device,
            honion_gpu::device_count().context("querying CUDA devices")?,
        )?
    };
    let listener = match &args.listen {
        Some(spec) => {
            let addr = crate::net::bind_addr(spec)?;
            let listener = std::net::TcpListener::bind(addr)
                .with_context(|| format!("listening for workers on {addr}"))?;
            if !args.quiet {
                eprintln!("honion: accepting workers on {addr}");
            }
            Some(listener)
        }
        None => None,
    };

    let difficulty = set.difficulty_log2() + suffixes.difficulty_log2();
    if !args.quiet {
        eprintln!(
            "honion: searching for {} pattern(s) on {} device(s){}",
            sourced.len(),
            devices.len(),
            if args.listen.is_some() {
                ", plus remote workers"
            } else {
                ""
            },
        );
        for s in &sourced {
            eprintln!(
                "  {} ({:.1} bits, from {})",
                s.pattern.source(),
                s.pattern.difficulty_log2(),
                s.origin
            );
        }
        for s in suffixes.patterns() {
            eprintln!(
                "  ends with {} ({:.1} bits, from --suffix)",
                s.source(),
                s.difficulty_log2()
            );
        }
        eprintln!(
            "  combined difficulty {difficulty:.1} bits, prefilter {:.1} bits",
            tables.prefilter_selectivity_log2()
        );
    }

    let progress = match &prior {
        Some(st) if args.resume => Progress::resumed(
            difficulty,
            args.quiet,
            std::time::Duration::from_secs_f64(st.elapsed_secs),
            st.examined,
            st.written as usize,
        ),
        _ => Progress::new(difficulty, args.quiet),
    };
    // What every checkpoint line must repeat; the counters come from the
    // shared progress at save time.
    let meta = SearchState {
        prefixes: sourced
            .iter()
            .map(|s| s.pattern.source().to_owned())
            .collect(),
        suffixes: suffix_srcs.clone(),
        elapsed_secs: 0.0,
        examined: 0,
        written: 0,
    };
    let shared = Arc::new(Shared {
        progress: Mutex::new(progress),
        written: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
        errors: Mutex::new(Vec::new()),
    });

    // One worker per device. Each owns its Searcher and its epoch's scalars,
    // which are drawn, uploaded and destroyed on that thread — every device
    // sees only the public keys of its own worker's scalars, so the
    // no-secret-on-the-device invariant of docs/05 holds per card unchanged.
    // The coordinator's connection threads are joined by handle rather
    // than by scope (scoped threads cannot spawn scoped threads), so they
    // take this context by Arc. The locals move by Arc too: the closure
    // below wants owned captures either way, and the clones are refcounts.
    let args = Arc::new(args.clone());
    let set = Arc::new(set);
    let suffixes = Arc::new(suffixes);
    let meta = Arc::new(meta);
    let coord_handle = listener
        .map(|listener| {
            crate::net::coord::serve(
                crate::net::coord::CoordCtx {
                    args: Arc::clone(&args),
                    set: Arc::clone(&set),
                    suffixes: Arc::clone(&suffixes),
                    meta: Arc::clone(&meta),
                    shared: Arc::clone(&shared),
                },
                listener,
            )
        })
        .transpose()?;

    std::thread::scope(|scope| {
        for &device in &devices {
            // Move a clone into each worker, not the shared state itself:
            // an `Arc` clone is a refcount bump, so every worker holds its
            // own pointer to the one Shared.
            let shared = Arc::clone(&shared);
            let set = Arc::clone(&set);
            let tables = &tables;
            let suffixes = Arc::clone(&suffixes);
            let meta = Arc::clone(&meta);
            let args = Arc::clone(&args);
            std::thread::Builder::new()
                .name(format!("honion-gpu{device}"))
                // NVRTC compiles the kernel and the driver JITs the PTX on
                // this thread now, not the main one. That recursion is deep
                // enough to overflow the 2 MiB default a spawned thread
                // gets; the main thread's 8 MiB never noticed.
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, move || {
                    if let Err(e) =
                        search_device(device, &args, &set, tables, &suffixes, &meta, &shared)
                    {
                        // Report the whole context chain, then take the
                        // other workers down at their next launch boundary
                        // rather than letting them search for a result that
                        // will not be wanted.
                        eprintln!("error: device {device}: {e:?}");
                        shared
                            .errors
                            .lock()
                            .expect("the error list was poisoned")
                            .push(format!("device {device}"));
                        shared.stop.store(true, Ordering::Relaxed);
                    }
                })
                .expect("spawning a search worker");
        }
    });

    // Every remote round is verified-or-dropped before the summary: the
    // overseer exits on `stop` and joins its connection threads, each of
    // which bounds its own socket waits.
    if let Some(handle) = coord_handle {
        handle
            .join()
            .map_err(|_| anyhow!("the coordinator thread panicked"))?;
    }

    let written = shared.written.load(Ordering::SeqCst);
    shared
        .progress
        .lock()
        .expect("the progress counter was poisoned")
        .report();

    let errors = shared.errors.lock().expect("the error list was poisoned");
    if !errors.is_empty() {
        // Devices and remote workers alike land in the error list; name the
        // failures rather than dividing by a worker count the coordinator
        // only half knows (remotes join and leave on their own schedule).
        bail!(
            "{} worker(s) failed: {} ({} key(s) written)",
            errors.len(),
            errors.join(", "),
            written,
        );
    }
    if !args.quiet {
        let progress = shared
            .progress
            .lock()
            .expect("the progress counter was poisoned");
        eprintln!(
            "honion: wrote {written} key(s); the search reported {} candidate(s) \
             while examining {} addresses in {}",
            progress.found(),
            si(progress.examined() as f64),
            humantime(progress.elapsed().as_secs_f64()),
        );
    }
    Ok(())
}

/// Search on one device until the run is over.
///
/// The loop is the single-device search unchanged — same launch pacing, same
/// host preparation overlapped with device work — with three additions:
/// `--count` is decided by the shared counter rather than a local one, the
/// shared `stop` flag is checked at launch boundaries, and progress is
/// recorded into the combined counter. A run with `--device 0` is this loop
/// and nothing else.
fn search_device(
    device: u32,
    args: &SearchArgs,
    set: &PatternSet,
    tables: &DeviceTables,
    suffixes: &SuffixSet,
    meta: &SearchState,
    shared: &Shared,
) -> Result<()> {
    // Sized to the card unless the user pinned it: throughput rises with
    // concurrent walks until their local memory stops fitting, so the right
    // number is a property of the device, not a constant.
    let threads = match args.threads {
        Some(0) => bail!("--threads must be positive"),
        Some(n) => n,
        None => honion_gpu::auto_threads(device, args.offsets)
            .with_context(|| format!("sizing the search to device {device}"))?,
    };

    // The hit buffer is sized for a very unlucky launch. Overflow is reported
    // rather than silent, but it should never happen.
    let max_hits = 4096u32;
    let mut searcher = Searcher::new(device, tables, threads, args.offsets, max_hits)
        .with_context(|| format!("preparing the GPU search on device {device}"))?;
    if !args.quiet {
        // Cosmetic, so a failed query degrades to a placeholder instead of
        // aborting a search that is otherwise working -- and backend-neutral,
        // so the Metal backend describes itself the same way.
        let info = searcher
            .device_info()
            .unwrap_or_else(|_| honion_gpu::DeviceInfo::unknown());
        eprintln!(
            "  device {device}: {info}, {} threads, \
             {} offsets ({} candidates per inversion, {:.1} GB device memory)",
            searcher.num_threads(),
            searcher.half(),
            searcher.candidates_per_batch(),
            honion_gpu::local_bytes_per_thread(searcher.half()) as f64
                * f64::from(searcher.num_threads())
                / 1e9
        );
    }

    // Start small so the first launch cannot run for an unexpectedly long time
    // on a slow device; the rate measured from it sizes every later launch.
    let mut candidates: u32 = 4096;

    // Draw the first epoch's scalars. They live only here, on this thread, in
    // host memory; the device receives their public keys and nothing else.
    let mut scalars = draw_scalars(threads)?;
    searcher
        .set_start_points(&public_keys(&scalars))
        .context("uploading starting points")?;

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let launch_start = std::time::Instant::now();
        let batches = searcher
            .launch_async(candidates)
            .context("running the search kernel")?;

        // The GPU is busy; draw the next epoch's scalars while it works. On a
        // large card this is hundreds of milliseconds that would otherwise be
        // dead time before every launch.
        let next_scalars = draw_scalars(threads)?;
        let next_points = public_keys(&next_scalars);

        let outcome = searcher
            .collect(batches)
            .context("collecting search results")?;
        let launch_time = launch_start.elapsed().as_secs_f64();

        if outcome.total_found as usize > outcome.hits.len() {
            eprintln!(
                "warning: device {device} found {} candidates but only {} fit the buffer; \
                 {} were dropped. Raise --min-selectivity or use a longer pattern.",
                outcome.total_found,
                outcome.hits.len(),
                outcome.total_found as usize - outcome.hits.len()
            );
        }

        // Verify every candidate against the host's own arithmetic before it
        // is allowed to become a file. Each worker checks its own device's
        // claims against its own scalars; one device's hit can never be
        // reconstructed from another's starting point.
        // Claims whose prefix matched but whose suffix did not. These are
        // expected noise — the suffix is priced into the difficulty and
        // costs this claim its write, nothing more — but they must not be
        // counted as found, or P(found) would claim keys that were never
        // written.
        let mut suffix_skips = 0usize;
        for hit in &outcome.hits {
            let Some(start) = scalars.get(hit.thread_id as usize) else {
                bail!(
                    "device reported thread {} but only {} were launched",
                    hit.thread_id,
                    scalars.len()
                );
            };
            let verified =
                VerifiedKey::verify(start, i64::from(hit.offset), set).with_context(|| {
                    format!(
                        "verifying the candidate reported at thread {} offset {}. \
                         This means the GPU and the CPU disagree, which is a bug or a \
                         hardware fault, not a user error.",
                        hit.thread_id, hit.offset
                    )
                })?;

            if !suffixes.matches_any(verified.address().body()) {
                suffix_skips += 1;
                continue;
            }

            let dir = match write_service_dir(&args.out, &verified) {
                Ok(dir) => dir,
                Err(StoreError::AlreadyExists { .. }) => {
                    // The memoryless search can hand us a key whose
                    // directory a previous run wrote; a resumed run will
                    // find its own hits again at low probability. Skip and
                    // keep searching -- the key exists on disk already,
                    // which is what --count ultimately counts.
                    if !args.quiet {
                        eprintln!(
                            "  {} already exists; skipping",
                            verified.address().to_hostname()
                        );
                    }
                    continue;
                }
                Err(e) => {
                    return Err(e).context("writing the hidden service directory");
                }
            };
            let total = shared.written.fetch_add(1, Ordering::SeqCst) + 1;
            println!("{}", verified.address().to_hostname());
            if !args.quiet {
                eprintln!("  written to {}", dir.display());
            }
            if args.count != 0 && total >= args.count {
                // Every key found up to and including this one is written —
                // a found key is never thrown away to honour a count. Two
                // devices finishing the same instant can therefore overshoot
                // `--count` by a handful, which is the only way the total
                // can exceed it.
                shared
                    .progress
                    .lock()
                    .expect("the progress counter was poisoned")
                    .record(
                        outcome.examined,
                        outcome.hits.len().saturating_sub(suffix_skips),
                    );
                let progress = shared
                    .progress
                    .lock()
                    .expect("the progress counter was poisoned");
                save_checkpoint(args, meta, &progress);
                shared.stop.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }

        {
            let mut progress = shared
                .progress
                .lock()
                .expect("the progress counter was poisoned");
            progress.record(
                outcome.examined,
                outcome.hits.len().saturating_sub(suffix_skips),
            );
            progress.report();
            save_checkpoint(args, meta, &progress);
        }

        // Hand the prepared epoch to the device and keep its scalars, which
        // the next round's hits will be reconstructed from.
        searcher
            .set_start_points(&next_points)
            .context("uploading starting points")?;
        scalars = next_scalars;

        // Size the next launch from the rate just measured, so the loop
        // settles on the requested wall-clock interval regardless of how
        // many devices are competing for the host's CPU.
        if launch_time > 0.0 {
            let per_candidate = launch_time / f64::from(candidates);
            let want = (args.launch_seconds / per_candidate).round();
            candidates = want.clamp(1.0, f64::from(u32::MAX / 2)) as u32;
        }
    }
}

/// Write the checkpoint for `--out`. Failures warn and never stop a
/// running search: the statistics are a courtesy, the keys on disk are the
/// product, and killing a working two-week run over an unreadable directory
/// would be the checkpoint causing the loss it exists to prevent.
pub(crate) fn save_checkpoint(args: &SearchArgs, meta: &SearchState, progress: &Progress) {
    let state = SearchState {
        prefixes: meta.prefixes.clone(),
        suffixes: meta.suffixes.clone(),
        elapsed_secs: progress.elapsed().as_secs_f64(),
        examined: progress.examined(),
        // Progress is seeded from the checkpoint on --resume, so its
        // count is cumulative across runs; `shared.written` is only this
        // process's, which would make a resumed checkpoint *forget* the
        // keys earlier runs wrote.
        written: progress.found() as u64,
    };
    if let Err(e) = state.save(&args.out, &args.device) {
        if !args.quiet {
            eprintln!("honion: could not write the checkpoint: {e:#}");
        }
    }
}

/// Parse `--suffix` arguments. Same discipline as prefixes (langsec rule 2):
/// all are recognised before any search starts, and the first malformed or
/// structurally impossible ending aborts the run.
fn parse_suffixes(args: &[String]) -> Result<SuffixSet> {
    let mut out = Vec::with_capacity(args.len());
    for src in args {
        out.push(SuffixPattern::parse(src).with_context(|| format!("--suffix {src:?}"))?);
    }
    Ok(SuffixSet::new(out))
}

/// Compile a pattern set from sourced patterns.
fn compile(sourced: &[SourcedPattern]) -> Result<PatternSet> {
    let list: Vec<Pattern> = sourced.iter().map(|s| s.pattern.clone()).collect();
    PatternSet::compile(&list).context("compiling patterns")
}

/// `honion estimate`.
///
/// # Errors
///
/// If patterns are malformed.
pub fn run_estimate(args: &EstimateArgs) -> Result<()> {
    let sourced = collect(&args.prefixes, args.patterns_file.as_ref())?;
    let set = compile(&sourced)?;
    let tables = DeviceTables::build(&set);
    let suffixes = parse_suffixes(&args.suffixes)?;

    println!("patterns: {}", sourced.len());
    for s in &sourced {
        println!(
            "  {:<24} {:>6.1} bits  expected {:>10}   [{}]",
            s.pattern.source(),
            s.pattern.difficulty_log2(),
            humantime(2f64.powf(s.pattern.difficulty_log2()) / args.rate),
            s.origin,
        );
    }
    if !suffixes.patterns().is_empty() {
        println!("endings: {}", suffixes.patterns().len());
        for s in suffixes.patterns() {
            println!(
                "  {:<24} {:>6.1} bits  expected {:>10}   [--suffix]",
                s.source(),
                s.difficulty_log2(),
                humantime(2f64.powf(s.difficulty_log2()) / args.rate),
            );
        }
    }
    let d = set.difficulty_log2() + suffixes.difficulty_log2();
    println!();
    println!(
        "combined difficulty : {d:.1} bits ({} expected trials)",
        si(2f64.powf(d))
    );
    println!("assumed rate        : {}/s", si(args.rate));
    println!(
        "expected time       : {}",
        humantime(2f64.powf(d) / args.rate)
    );
    println!(
        "prefilter           : {:.1} bits of selectivity",
        tables.prefilter_selectivity_log2()
    );
    println!();
    println!("The search is memoryless: each key is an independent trial, so the");
    println!("expected time is a mean, not a deadline. There is a 63% chance of a");
    println!("result by then, 86% by twice that, and 95% by three times.");
    Ok(())
}

/// Draw one clamped secret scalar per thread from the system CSPRNG.
///
/// The buffer is locked against swap on the way out (see `secure_mem`); the
/// scalars themselves zeroize on drop and never leave this process's CPU
/// side, so this closes the last window docs/05 left open.
pub(crate) fn draw_scalars(threads: u32) -> Result<Vec<SecretScalar>> {
    let scalars = (0..threads)
        .into_par_iter()
        .map(|_| SecretScalar::generate())
        .collect::<Result<Vec<_>, _>>()
        .context("drawing secret scalars from the system CSPRNG")?;
    secure_mem::lock_secret_buffer(&scalars);
    Ok(scalars)
}

/// The public keys of those scalars. Only these reach the device.
pub(crate) fn public_keys(scalars: &[SecretScalar]) -> Vec<[u8; 32]> {
    scalars.par_iter().map(SecretScalar::public_key).collect()
}

/// `honion verify`.
///
/// Re-derives the address from the stored secret key and checks it against the
/// stored `hostname`, so a directory can be checked long after it was made —
/// including one produced by other tools.
///
/// # Errors
///
/// If the directory is incomplete, malformed, or internally inconsistent.
pub fn run_verify(dir: &Path) -> Result<()> {
    let secret_path = dir.join("hs_ed25519_secret_key");
    let hostname_path = dir.join("hostname");

    let secret = std::fs::read(&secret_path)
        .with_context(|| format!("reading {}", secret_path.display()))?;
    if secret.len() != 96 {
        bail!(
            "{} is {} bytes, expected 96 (a 32-byte tag and a 64-byte expanded key)",
            secret_path.display(),
            secret.len()
        );
    }
    let scalar_bytes: [u8; 32] = secret[32..64].try_into().expect("slice is 32 bytes");
    let scalar = SecretScalar::from_clamped(scalar_bytes)
        .context("the stored scalar is not in clamped form")?;

    let public = scalar.public_key();
    let derived = OnionAddress::from_pubkey(&public);

    let stored = std::fs::read_to_string(&hostname_path)
        .with_context(|| format!("reading {}", hostname_path.display()))?;
    let stored = stored.trim();
    if stored != derived.to_hostname() {
        bail!(
            "mismatch: {} says {stored}, but the secret key derives {}",
            hostname_path.display(),
            derived.to_hostname()
        );
    }

    // Confirm the public key file agrees too, if present.
    let public_path = dir.join("hs_ed25519_public_key");
    if public_path.exists() {
        let bytes = std::fs::read(&public_path)
            .with_context(|| format!("reading {}", public_path.display()))?;
        if bytes.len() != 64 || bytes[32..] != public[..] {
            bail!(
                "{} does not contain the public key implied by the secret key",
                public_path.display()
            );
        }
    }

    println!("{} verified: {}", dir.display(), derived.to_hostname());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_devices;

    #[test]
    fn all_expands_to_every_visible_device() {
        assert_eq!(parse_devices("all", 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_devices("ALL", 1).unwrap(), vec![0]);
    }

    #[test]
    fn lists_keep_their_order_and_allow_repeats() {
        assert_eq!(parse_devices("0", 2).unwrap(), vec![0]);
        assert_eq!(parse_devices("2,0", 4).unwrap(), vec![2, 0]);
        assert_eq!(parse_devices(" 1 , 1 ", 2).unwrap(), vec![1, 1]);
    }

    #[test]
    fn malformed_specs_are_refused() {
        assert!(parse_devices("", 2).is_err());
        assert!(parse_devices("0,,1", 2).is_err());
        assert!(parse_devices("x", 2).is_err());
        // A machine with no devices cannot satisfy any request, however
        // well-formed.
        assert!(parse_devices("all", 0).is_err());
        assert!(parse_devices("0", 0).is_err());
    }

    #[test]
    fn ordinals_beyond_the_machine_are_named_in_the_error() {
        let err = parse_devices("3", 2).unwrap_err().to_string();
        assert!(err.contains("no device 3"), "unhelpful: {err}");
        assert!(err.contains("0..1"), "should state the valid range: {err}");
    }
}
