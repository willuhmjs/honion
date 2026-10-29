//! The frame format: seven message kinds, auditable in one screen.
//!
//! Every reader validates before it trusts: a fixed magic, a one-byte
//! type from a closed set, a length ceiling applied *before* any
//! allocation, UTF-8 that must actually be UTF-8, and start-point
//! payloads that must be an exact multiple of 32 bytes. A peer speaking
//! nonsense — including one deliberately trying — can only ever cause a
//! clean [`ProtoError`], never a panic and never an unbounded read.
//!
//! Numbers are big-endian on the wire. Integers keep the widths they
//! have in the rest of the codebase (`u32` threads, `i32` offsets), so
//! no conversion can silently truncate a value the other end will
//! misread.

use std::io::{Read, Write};

use thiserror::Error;

/// `"HON1"`: this protocol, at this version. The magic *is* the version:
/// any layout change bumps the digit, and a peer that speaks another is
/// not honion, so the connection is dead on arrival.
pub const MAGIC: u32 = 0x484F_4E31;

/// Hard ceiling on any single payload, enforced before allocating, so a
/// hostile or confused peer cannot steer the reader into an
/// out-of-memory death. Legitimate frames stay far below it: a full
/// epoch of start points for a 4-million-thread launch is ~128 MB of
/// points plus header, and 64 MiB bounds epochs at two million threads.
pub const MAX_PAYLOAD: u32 = 64 * 1024 * 1024;

/// Ceiling on one string field inside a payload; patterns are at most
/// 56 characters, so this only ever fires on nonsense.
const MAX_STRING: u32 = 1 << 20;

/// Why a connection is not usable. Every variant is the peer's fault,
/// and the only correct response at any call site is to close the
/// connection and report.
#[derive(Debug, Error)]
pub enum ProtoError {
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic {0:#010x}: the peer is not honion's search protocol")]
    BadMagic(u32),
    #[error("unknown message type {0}")]
    BadType(u8),
    #[error("payload length {0} exceeds the {MAX_PAYLOAD} byte ceiling")]
    Oversize(u32),
    #[error("payload ended mid-value")]
    Truncated,
    #[error("string field is not valid UTF-8")]
    BadUtf8,
    #[error("start points are {0} bytes, not a multiple of 32")]
    BadPoints(usize),
    #[error("too many {kind} for one message: {n}")]
    AbsurdCount {
        /// What was being counted.
        kind: &'static str,
        /// The count the peer claimed.
        n: u32,
    },
}

/// One protocol message.
#[derive(Clone, Debug, PartialEq)]
pub enum Msg {
    /// Worker's opening line, before it knows what the search even is.
    Hello {
        /// Human-readable device description, for the coordinator's log.
        device: String,
    },
    /// Coordinator's answer: everything needed to build the exact same
    /// device tables the coordinator would build for its own GPUs.
    Config {
        /// Prefix pattern sources; the worker compiles them itself.
        patterns: Vec<String>,
        /// Suffix sources, likewise validated and re-parsed by the worker.
        suffixes: Vec<String>,
        /// Offsets per batch (`--offsets`), i.e. the kernel's `HALF`.
        half: u32,
        /// Target wall time for one launch; the worker sizes batches to
        /// report this often.
        launch_seconds: f64,
        /// Prefilter selectivity floor; the worker refuses the same weak
        /// patterns the coordinator would refuse for itself.
        min_selectivity: f64,
    },
    /// One epoch: the public halves of freshly drawn scalars, `threads`
    /// of 32-byte compressed curve points. The scalars stay home.
    Epoch {
        /// Monotonic per connection; reports must name the epoch they
        /// answer so a stale or forged report is recognisable.
        id: u64,
        /// Public start points, one per thread.
        points: Vec<[u8; 32]>,
    },
    /// A worker's answer to an epoch: what it examined and claims the
    /// coordinator must verify itself. Claims are *claims*.
    Report {
        /// The epoch being answered.
        id: u64,
        /// Candidates examined; a trust-for-display number, not a
        /// security-relevant one.
        examined: u64,
        /// `(thread, offset)` pairs the worker's device flagged.
        claims: Vec<(u32, i32)>,
    },
    /// The worker has built its searcher: here is the thread count every
    /// epoch must match. Follows [`Msg::Config`] because sizing threads
    /// needs the batch width only the config carries.
    Ready {
        /// Threads the worker will search per epoch.
        threads: u32,
    },
    /// The search is over (`--count` met or the operator stopped it);
    /// the worker may exit.
    Done,
    /// A refusal with a reason, sent before hanging up.
    Error(String),
}

// Message type bytes. The numbering is part of the wire format: adding
// a kind means a new number, and changing any existing payload's layout
// means a new magic digit.
const T_HELLO: u8 = 1;
const T_CONFIG: u8 = 2;
const T_EPOCH: u8 = 3;
const T_REPORT: u8 = 4;
const T_DONE: u8 = 5;
const T_ERROR: u8 = 6;
const T_READY: u8 = 7;

/// Write one message: envelope header, then payload.
///
/// # Errors
///
/// On any I/O failure. A half-written frame closes the connection by
/// contract; readers never see it because the writer owns the socket
/// and its failure propagates to the owning loop.
pub fn write_msg(w: &mut impl Write, msg: &Msg) -> Result<(), ProtoError> {
    let (kind, payload) = match msg {
        Msg::Hello { device } => {
            let mut p = Vec::new();
            put_string(&mut p, device);
            (T_HELLO, p)
        }
        Msg::Config {
            patterns,
            suffixes,
            half,
            launch_seconds,
            min_selectivity,
        } => {
            let mut p = Vec::new();
            put_strings(&mut p, patterns);
            put_strings(&mut p, suffixes);
            put_u32(&mut p, *half);
            p.extend_from_slice(&min_selectivity.to_bits().to_be_bytes());
            // f64 crosses the wire bit-exactly; both ends agree on
            // IEEE 754 by construction, no locale or text round-trip.
            p.extend_from_slice(&launch_seconds.to_bits().to_be_bytes());
            (T_CONFIG, p)
        }
        Msg::Epoch { id, points } => {
            let mut p = Vec::with_capacity(12 + points.len() * 32);
            p.extend_from_slice(&id.to_be_bytes());
            put_u32(&mut p, points.len() as u32);
            for pt in points {
                p.extend_from_slice(pt);
            }
            (T_EPOCH, p)
        }
        Msg::Report {
            id,
            examined,
            claims,
        } => {
            let mut p = Vec::with_capacity(20 + claims.len() * 8);
            p.extend_from_slice(&id.to_be_bytes());
            p.extend_from_slice(&examined.to_be_bytes());
            put_u32(&mut p, claims.len() as u32);
            for &(thread, offset) in claims {
                put_u32(&mut p, thread);
                p.extend_from_slice(&offset.to_be_bytes());
            }
            (T_REPORT, p)
        }
        Msg::Ready { threads } => {
            let mut p = Vec::new();
            put_u32(&mut p, *threads);
            (T_READY, p)
        }
        Msg::Done => (T_DONE, Vec::new()),
        Msg::Error(m) => {
            let mut p = Vec::new();
            put_string(&mut p, m);
            (T_ERROR, p)
        }
    };
    if payload.len() as u64 > u64::from(MAX_PAYLOAD) {
        // A coordinator with an absurd --threads could ask for a larger
        // epoch than the ceiling allows; refuse to send rather than
        // invent a frame format on the fly.
        return Err(ProtoError::Oversize(payload.len() as u32));
    }
    let mut header = [0u8; 9];
    header[..4].copy_from_slice(&MAGIC.to_be_bytes());
    header[4] = kind;
    header[5..].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    w.write_all(&header)?;
    w.write_all(&payload)?;
    w.flush()?;
    Ok(())
}

/// Read one message, validating the whole envelope and payload.
///
/// # Errors
///
/// [`ProtoError`] on any protocol violation or I/O failure; see its
/// variants. Never panics on adversarial input.
pub fn read_msg(r: &mut impl Read) -> Result<Msg, ProtoError> {
    let mut header = [0u8; 9];
    r.read_exact(&mut header)?;
    let magic = u32::from_be_bytes(header[..4].try_into().unwrap_or([0; 4]));
    if magic != MAGIC {
        return Err(ProtoError::BadMagic(magic));
    }
    let kind = header[4];
    let len = u32::from_be_bytes(header[5..].try_into().unwrap_or([0; 4]));
    if len > MAX_PAYLOAD {
        return Err(ProtoError::Oversize(len));
    }
    let mut p = vec![0u8; len as usize];
    r.read_exact(&mut p)?;
    let mut cur = 0usize;
    match kind {
        T_HELLO => {
            let device = take_string(&p, &mut cur)?;
            Ok(Msg::Hello { device })
        }
        T_CONFIG => {
            let patterns = take_strings(&p, &mut cur)?;
            let suffixes = take_strings(&p, &mut cur)?;
            let half = take_u32(&p, &mut cur)?;
            let min_selectivity = f64::from_bits(take_u64(&p, &mut cur)?);
            let launch_seconds = f64::from_bits(take_u64(&p, &mut cur)?);
            Ok(Msg::Config {
                patterns,
                suffixes,
                half,
                launch_seconds,
                min_selectivity,
            })
        }
        T_EPOCH => {
            let id = take_u64(&p, &mut cur)?;
            let n = take_u32(&p, &mut cur)?;
            let rest = p.len() - cur;
            if rest % 32 != 0 || rest < usize::try_from(n).unwrap_or(usize::MAX) * 32 {
                return Err(ProtoError::BadPoints(rest));
            }
            let mut points = Vec::with_capacity(n.min(MAX_PAYLOAD / 32) as usize);
            for _ in 0..n {
                let mut pt = [0u8; 32];
                pt.copy_from_slice(&p[cur..cur + 32]);
                cur += 32;
                points.push(pt);
            }
            if cur != p.len() {
                return Err(ProtoError::BadPoints(p.len() - cur));
            }
            Ok(Msg::Epoch { id, points })
        }
        T_REPORT => {
            let id = take_u64(&p, &mut cur)?;
            let examined = take_u64(&p, &mut cur)?;
            let n = take_u32(&p, &mut cur)?;
            if usize::try_from(n).unwrap_or(usize::MAX) > (p.len() - cur) / 8 {
                return Err(ProtoError::AbsurdCount { kind: "claims", n });
            }
            let mut claims = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let thread = take_u32(&p, &mut cur)?;
                let offset = i32::from_be_bytes(take_u32(&p, &mut cur)?.to_be_bytes());
                claims.push((thread, offset));
            }
            Ok(Msg::Report {
                id,
                examined,
                claims,
            })
        }
        T_READY => Ok(Msg::Ready {
            threads: take_u32(&p, &mut cur)?,
        }),
        T_DONE => Ok(Msg::Done),
        T_ERROR => Ok(Msg::Error(take_string(&p, &mut cur)?)),
        other => Err(ProtoError::BadType(other)),
    }
}

// --- payload primitives ---------------------------------------------------

fn put_u32(p: &mut Vec<u8>, v: u32) {
    p.extend_from_slice(&v.to_be_bytes());
}

fn put_string(p: &mut Vec<u8>, s: &str) {
    put_u32(p, s.len() as u32);
    p.extend_from_slice(s.as_bytes());
}

fn put_strings(p: &mut Vec<u8>, v: &[String]) {
    put_u32(p, v.len() as u32);
    for s in v {
        put_string(p, s);
    }
}

fn take_u32(p: &[u8], cur: &mut usize) -> Result<u32, ProtoError> {
    let end = cur.checked_add(4).ok_or(ProtoError::Truncated)?;
    let bytes: [u8; 4] = p
        .get(*cur..end)
        .ok_or(ProtoError::Truncated)?
        .try_into()
        .unwrap_or([0; 4]);
    *cur = end;
    Ok(u32::from_be_bytes(bytes))
}

fn take_u64(p: &[u8], cur: &mut usize) -> Result<u64, ProtoError> {
    let end = cur.checked_add(8).ok_or(ProtoError::Truncated)?;
    let bytes: [u8; 8] = p
        .get(*cur..end)
        .ok_or(ProtoError::Truncated)?
        .try_into()
        .unwrap_or([0; 8]);
    *cur = end;
    Ok(u64::from_be_bytes(bytes))
}

fn take_string(p: &[u8], cur: &mut usize) -> Result<String, ProtoError> {
    let n = take_u32(p, cur)?;
    if n > MAX_STRING {
        return Err(ProtoError::AbsurdCount { kind: "string", n });
    }
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    let end = cur.checked_add(n).ok_or(ProtoError::BadUtf8)?;
    let bytes = p.get(*cur..end).ok_or(ProtoError::BadUtf8)?;
    *cur = end;
    String::from_utf8(bytes.to_vec()).map_err(|_| ProtoError::BadUtf8)
}

fn take_strings(p: &[u8], cur: &mut usize) -> Result<Vec<String>, ProtoError> {
    let n = take_u32(p, cur)?;
    if n > MAX_STRING / 4 {
        return Err(ProtoError::AbsurdCount { kind: "strings", n });
    }
    let mut out = Vec::new();
    for _ in 0..n {
        out.push(take_string(p, cur)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(msg: &Msg) {
        let mut buf = Vec::new();
        write_msg(&mut buf, msg).expect("write");
        let got = read_msg(&mut buf.as_slice()).expect("read");
        assert_eq!(&got, msg);
    }

    #[test]
    fn every_message_round_trips() {
        round_trip(&Msg::Hello {
            device: "NVIDIA GeForce RTX 4090 (CUDA, compute capability 8.9)".into(),
        });
        round_trip(&Msg::Ready { threads: 177_152 });
        round_trip(&Msg::Config {
            patterns: vec!["carroll".into(), "hon?on[abc]".into()],
            suffixes: vec!["qad".into()],
            half: 512,
            launch_seconds: 4.25,
            min_selectivity: 20.0,
        });
        round_trip(&Msg::Epoch {
            id: u64::MAX,
            points: vec![[7u8; 32], [0u8; 32]],
        });
        round_trip(&Msg::Epoch {
            id: 0,
            points: Vec::new(),
        });
        round_trip(&Msg::Report {
            id: 3,
            examined: 12_884_901_888,
            claims: vec![(0, -17), (u32::MAX, i32::MIN), (5, i32::MAX)],
        });
        round_trip(&Msg::Done);
        round_trip(&Msg::Error("no patterns".into()));
    }

    #[test]
    fn foreign_magic_kills_the_connection() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        buf.push(T_DONE);
        buf.extend_from_slice(&0u32.to_be_bytes());
        assert!(matches!(
            read_msg(&mut buf.as_slice()),
            Err(ProtoError::BadMagic(0xDEAD_BEEF))
        ));
    }

    #[test]
    fn oversized_length_is_refused_before_allocating() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(T_EPOCH);
        buf.extend_from_slice(&MAX_PAYLOAD.to_be_bytes()); // one over would be >; use max+1
        // Rewrite length to MAX_PAYLOAD + 1 by hand:
        buf[5..].copy_from_slice(&MAX_PAYLOAD.to_be_bytes());
        let mut huge = buf.clone();
        huge[5..].copy_from_slice(&(MAX_PAYLOAD + 1).to_be_bytes());
        // exactly at the ceiling is legal to *attempt* (the payload read
        // would then hit EOF); one over is refused immediately:
        assert!(matches!(
            read_msg(&mut huge.as_slice()),
            Err(ProtoError::Oversize(_))
        ));
        // the at-ceiling frame fails on I/O (EOF), never allocation panic:
        let _ = read_msg(&mut buf.as_slice());
    }

    #[test]
    fn truncated_and_bogus_payloads_are_errors_not_panics() {
        // Truncated hello: claims a huge string but sends nothing.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(T_HELLO);
        buf.extend_from_slice(&8u32.to_be_bytes());
        buf.extend_from_slice(&177_152u32.to_be_bytes());
        buf.extend_from_slice(&0xFFFF_FFF0u32.to_be_bytes()); // string length
        assert!(read_msg(&mut buf.as_slice()).is_err());

        // Invalid UTF-8 in a device name.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(T_HELLO);
        let payload: Vec<u8> = 1u32.to_be_bytes().into_iter().chain([0xFF]).collect();
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&payload);
        assert!(matches!(
            read_msg(&mut buf.as_slice()),
            Err(ProtoError::BadUtf8)
        ));

        // Unknown type.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(99);
        buf.extend_from_slice(&0u32.to_be_bytes());
        assert!(matches!(
            read_msg(&mut buf.as_slice()),
            Err(ProtoError::BadType(99))
        ));

        // Epoch points not a multiple of 32.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(T_EPOCH);
        let payload: Vec<u8> = 1u64
            .to_be_bytes()
            .into_iter()
            .chain(1u32.to_be_bytes())
            .chain([0u8; 31])
            .collect();
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&payload);
        assert!(matches!(
            read_msg(&mut buf.as_slice()),
            Err(ProtoError::BadPoints(_))
        ));

        // Report claiming more claims than the payload could hold.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_be_bytes());
        buf.push(T_REPORT);
        let payload: Vec<u8> = 1u64
            .to_be_bytes()
            .into_iter()
            .chain(0u64.to_be_bytes())
            .chain(1000u32.to_be_bytes()) // 1000 claims...
            .chain([0u8; 8]) // ...with room for one
            .collect();
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&payload);
        assert!(matches!(
            read_msg(&mut buf.as_slice()),
            Err(ProtoError::AbsurdCount { .. })
        ));
    }

    #[test]
    fn offsets_survive_the_wire_with_their_sign() {
        // i32 crosses as u32 bits; MIN must arrive as MIN, not clamped.
        let msg = Msg::Report {
            id: 1,
            examined: 0,
            claims: vec![(1, i32::MIN), (2, i32::MAX), (3, -1)],
        };
        let mut buf = Vec::new();
        write_msg(&mut buf, &msg).expect("write");
        let got = read_msg(&mut buf.as_slice()).expect("read");
        assert_eq!(msg, got);
    }
}
