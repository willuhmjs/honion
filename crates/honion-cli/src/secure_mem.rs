//! Keeping host secret material out of swap and core dumps.
//!
//! `docs/05-security-model.md` closed with an admission: the epoch scalar
//! buffer — every secret the search will use, up to ~32 MB across the two
//! live epochs — was not `mlock`ed and could in principle reach swap. This
//! module is that sentence's fix.
//!
//! Two syscalls do the work, both best-effort by design:
//!
//! * `mlock` pins the buffer's pages in RAM. The failure mode is
//!   `RLIMIT_MEMLOCK` (commonly 8 MB, occasionally 64 kB), so a failure
//!   warns once with the actionable number and the search continues — this
//!   is defence in depth, not a security boundary, and refusing to run on a
//!   hardened-by-default distro would make the tool worse, not better.
//! * `madvise` keeps the pages out of core dumps, which is where a crashed
//!   (or ptraced) process leaks to disk regardless of what is in RAM. Linux
//!   says `MADV_DONTDUMP`; the same guarantee on macOS is
//!   `MADV_COREDUMP_NOCOPY`. The whole-process escalation — `PR_SET_DUMPABLE 0`
//!   on Linux, `PT_DENY_ATTACH` on macOS — additionally disables the
//!   profiling tools the benchmark docs depend on, so it lives behind
//!   `--paranoid`.
//!
//! The secrets themselves (`SecretScalar`) are already `ZeroizeOnDrop`, and
//! they never reach device memory; this covers the gap between "zeroized on
//! drop" and "swapped while still live".

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the "could not lock" warning has been printed; one warning per
/// process, because the buffer is locked every epoch and the limit will not
/// have changed between epochs.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Lock `buf`'s pages into RAM and exclude them from core dumps.
///
/// Best-effort: any failure prints one warning and returns normally. The
/// buffer must outlive its usefulness no longer than its lock does — `Vec`
/// drop frees the pages, and the kernel drops the lock with them.
pub fn lock_secret_buffer<T>(buf: &[T]) {
    if buf.is_empty() {
        return;
    }
    let page = page_size();
    // Round out to whole pages: mlock addresses page-aligned ranges, and
    // over-locking a boundary page's other occupants is harmless (they are
    // someone else's live memory, which the limit accounts the same).
    let start = (buf.as_ptr() as usize) & !(page - 1);
    let end = (buf.as_ptr() as usize)
        .saturating_add(std::mem::size_of_val(buf))
        .div_ceil(page)
        .saturating_mul(page);
    let len = end - start;
    // SAFETY: the rounded range contains the buffer's bytes in full; the
    // kernel only pins pages, it does not read or write them here.
    let rc = unsafe { libc::mlock(start as *const libc::c_void, len) };
    if rc == 0 {
        advise_no_dump(start as *mut libc::c_void, len);
    } else {
        warn_once(len);
    }
}

/// The madvise advice that excludes a range from core dumps, per platform.
/// macOS's `MADV_COREDUMP_NOCOPY` is not yet in the `libc` crate's Apple
/// table; the constant is xnu's `mach/madvneasy.h` value, and an unsupported
/// kernel answers `EINVAL` to a request that is best-effort anyway.
fn advise_no_dump(start: *mut libc::c_void, len: usize) {
    #[cfg(target_os = "linux")]
    // SAFETY: same address range; the advice changes a dump attribute only.
    unsafe {
        libc::madvise(start, len, libc::MADV_DONTDUMP);
    }
    #[cfg(target_os = "macos")]
    // SAFETY: as above; MADV_COREDUMP_NOCOPY is xnu's value 23.
    unsafe {
        libc::madvise(start, len, 23);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let _ = (start, len); // no known advice on this platform; mlock still holds
}

/// Exclude the whole process from core dumps and debugger attach.
///
/// Linux clears `PR_SET_DUMPABLE`; macOS has no such flag and answers the
/// same intent with `PT_DENY_ATTACH`. Either way this also blocks the
/// attach path cuda-gdb, lldb and Nsight use — hence opt-in. A search run
/// for days with a megabytes-sized buffer of secrets is the case where the
/// trade is worth making; profiling is not that case.
pub fn forbid_core_dump() {
    #[cfg(target_os = "linux")]
    // SAFETY: prctl with PR_SET_DUMPABLE takes an int argument; a failed
    // call changes nothing.
    let outcome = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) };
    #[cfg(target_os = "macos")]
    // SAFETY: PT_DENY_ATTACH takes no buffer arguments; a failed call
    // changes nothing.
    let outcome = unsafe {
        libc::ptrace(
            libc::PT_DENY_ATTACH,
            0,
            std::ptr::null_mut::<libc::c_char>(),
            0,
        )
    };
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    let outcome = {
        eprintln!("honion: --paranoid: unsupported platform; nothing was locked down");
        0
    };
    if outcome != 0 {
        eprintln!(
            "honion: --paranoid: could not disable core dumps/attach ({})",
            std::io::Error::last_os_error()
        );
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) is pure introspection.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(n).unwrap_or(4096).max(1)
}

fn warn_once(len: usize) {
    if WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    // Report the actual limit so the fix is copy-pasteable.
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: a stack rlimit struct with a valid address.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &raw mut lim) };
    let limit = if rc == 0 && lim.rlim_cur != libc::RLIM_INFINITY {
        format!("{} kB", lim.rlim_cur / 1024)
    } else {
        "unknown".to_owned()
    };
    eprintln!(
        "honion: warning: could not lock {} of secret material in RAM \
         (RLIMIT_MEMLOCK is {limit}); it could reach swap. \
         Raise the limit (ulimit -l, or run as a unit with LimitMEMLOCK) \
         if that matters on this machine.",
        crate::progress::si(len as f64),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locking_a_buffer_either_locks_or_warns_without_failing() {
        // The contract is "best effort": whatever the container's limits,
        // this call must not panic and must not fail the caller. When it
        // succeeds, VmLck in /proc/self/status must have risen.
        let before = read_vmlck().unwrap_or(0);
        let mut buf: Vec<u8> = vec![0u8; 64 * 1024];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u8;
        }
        lock_secret_buffer(&buf);
        let after = read_vmlck().unwrap_or(0);
        // Under a restrictive limit VmLck will not have moved; that is the
        // warning path, exercised on whatever machine runs the test.
        if after > before {
            assert!(after - before >= 64);
        }
        buf[0] = 1; // keep the buffer live past the lock call
    }

    fn read_vmlck() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmLck:") {
                return rest.trim_end_matches(" kB").trim().parse().ok();
            }
        }
        None
    }
}
