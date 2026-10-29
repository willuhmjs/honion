//! Distributed search: a coordinator renting GPUs it does not trust.
//!
//! # What this adds, in one sentence
//!
//! The coordinator of [`crate::search`] draws every secret scalar and
//! derives every public start point; a remote worker sees *less* than a
//! local GPU does today — it never sees a scalar, never sees which claim
//! was real, and never writes a file — so the trust model of
//! `docs/05-security-model.md` extends across the network unchanged.
//!
//! # Wire shape
//!
//! A hand-rolled binary protocol over one TCP connection per worker
//! ([`proto`]): no new dependency, a frame format small enough to audit
//! in one screen, and every message validated on read (magic, version,
//! length ceilings, UTF-8, point multiples) because the peer may be
//! hostile even though the protocol assumes it is merely unreliable.
//!
//! Nothing secret rides the wire, and nothing needed to stay secret
//! rides it: pattern text is configuration the coordinator would
//! otherwise type at its own keyboard, and start points are public
//! by construction. What is *not* provided is transport authentication
//! or confidentiality — an eavesdropper learns the patterns (they are
//! not secret) and a man in the middle can waste time or forge claims,
//! but every claim is re-derived and re-verified on the coordinator
//! before a key file is written, so a forged claim fails exactly like a
//! flaky GPU's would. Run it across a network you trust not to waste
//! your time; that is the honest boundary.

pub mod coord;
pub mod proto;
pub mod worker;

/// The port `--listen` and `--coordinator` assume when given no explicit
/// `:port`, so that examples and defaults agree on both sides.
pub const DEFAULT_PORT: u16 = 8_121;

/// Resolve a `--listen` specification to a bind address.
///
/// Accepts `PORT` (bind every interface — the common case is a machine
/// whose whole job is serving workers), `HOST:PORT`, or `:PORT`. A bare
/// interface-less host binds everything too, because typing
/// `--listen example.com` and getting a localhost bind would be the
/// surprise version of the same request.
///
/// # Errors
///
/// If the text names no parseable address.
pub fn bind_addr(spec: &str) -> Result<std::net::SocketAddr, anyhow::Error> {
    use anyhow::Context as _;
    let with_host = if let Some(port) = spec.strip_prefix(':') {
        format!("0.0.0.0:{port}")
    } else if spec.contains(':') {
        spec.to_owned()
    } else {
        format!("0.0.0.0:{spec}")
    };
    with_host
        .parse()
        .with_context(|| format!("--listen {spec:?} is not an address or port"))
}

#[cfg(test)]
mod tests {
    use super::bind_addr;

    #[test]
    fn listen_specs_default_to_every_interface() {
        assert_eq!(bind_addr("8121").unwrap().ip().to_string(), "0.0.0.0");
        assert_eq!(bind_addr(":8121").unwrap().ip().to_string(), "0.0.0.0");
        assert_eq!(bind_addr("8121").unwrap().port(), 8_121);
        assert_eq!(bind_addr(":8121").unwrap().port(), 8_121);
    }

    #[test]
    fn explicit_host_is_honoured() {
        let addr = bind_addr("127.0.0.1:9000").unwrap();
        assert_eq!(addr.ip().to_string(), "127.0.0.1");
        assert_eq!(addr.port(), 9_000);
        // v6 needs the brackets, same as every other tool's address syntax.
        let addr = bind_addr("[::1]:9000").unwrap();
        assert_eq!(addr.ip().to_string(), "::1");
    }

    #[test]
    fn nonsense_is_refused_with_the_user_words() {
        assert!(bind_addr("").is_err());
        assert!(bind_addr("not a port").is_err());
        assert!(bind_addr("8121.2").is_err());
    }
}
