//! Suffix patterns: constraints anchored to the **end** of the address.
//!
//! # Which endings are satisfiable, and what they cost
//!
//! The address body is `base32(PUBKEY ‖ CHECKSUM ‖ VERSION)` — see
//! [`crate::address`] — so the last characters are *not* uniformly distributed
//! in the way public-key characters are. Exactly:
//!
//! | position (0-indexed) | content | cost of a literal | admissible characters |
//! |---|---|---|---|
//! | ≤ 50 | public key | 5 bits | all 32 (searched by the device) |
//! | 51 | x sign bit + checksum | 5 bits | all 32 |
//! | 52, 53 | checksum | 5 bits | all 32 |
//! | 54 | 2 checksum bits, then 3 fixed zeros | **2 bits** | `a`, `i`, `q`, `y` only |
//! | 55 | the low 5 bits of `VERSION` (`0x03`) | **0 bits** | `d`, always |
//!
//! So every real `.onion` address ends in `d`, and the character before it is
//! one of four. A four-character suffix therefore costs at most 12 bits, not
//! 20 — and a suffix like `lock` is *unsatisfiable*, because `c` cannot appear
//! at position 54 and `k` cannot appear at position 55. That is rejected at
//! parse time rather than searched forever.
//!
//! # Where suffixes are enforced
//!
//! Suffixes beyond position 51 depend on the SHA3-256 checksum, which the
//! search kernel does not compute — no cheap algebraic handle exists (see
//! `docs/00-onion-v3-address-spec.md`). A suffix search therefore runs on the
//! device's *prefix* selectivity and checks the suffix on the host, inside
//! the verification path that already computes the full checksum for every
//! candidate the device claims. The suffix costs no search throughput; it
//! only rejects claims that matched the prefix. Consequently a suffix alone
//! is not searchable (the device would forward every candidate), and the CLI
//! requires at least one prefix; only prefixes feed [`super::PatternSet`].
//!
//! Recognition still goes through the single grammar
//! ([`super::parse::Pattern::parse`]); this module only re-anchors the atom
//! positions and enforces the structural rules the table above implies.

use std::fmt;

use crate::address::ADDRESS_CHARS;
use crate::base32::Base32Char;

use super::CharClass;
use super::parse::{Pattern, PatternError};

/// Why a suffix cannot constrain anything useful, or cannot be satisfied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuffixError {
    /// The suffix text itself failed the pattern grammar.
    Parse(PatternError),
    /// The address always ends in `d` (the `VERSION` byte), so a suffix
    /// naming anything else at the final position can never match.
    NotTerminalCharacter {
        /// The character the suffix demands at the final position.
        found: char,
    },
    /// Position 54 carries only two checksum bits, so only `a`, `i`, `q` and
    /// `y` can ever appear there; the named character cannot.
    ImpossibleChecksumCharacter {
        /// The 0-indexed address position (52..=55) of the offending atom.
        position: usize,
        /// The impossible character.
        found: char,
    },
}

impl fmt::Display for SuffixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "{e}"),
            Self::NotTerminalCharacter { found } => write!(
                f,
                "every onion address ends in 'd' (it is a fixed part of the address, not \
                 searchable material); the suffix demands '{found}' at the final position"
            ),
            Self::ImpossibleChecksumCharacter { position, found } => write!(
                f,
                "address position {position} can only ever hold one of a, i, q, y (its two \
                 variable bits are part of the checksum); the suffix demands '{found}'"
            ),
        }
    }
}

impl std::error::Error for SuffixError {}

impl From<PatternError> for SuffixError {
    fn from(e: PatternError) -> Self {
        Self::Parse(e)
    }
}

/// A parsed, anchored, structurally-checked suffix constraint.
///
/// Constructed only by [`SuffixPattern::parse`].
#[derive(Clone, Debug)]
pub struct SuffixPattern {
    /// The per-position character classes, innermost-last.
    classes: Vec<CharClass>,
    /// 0-indexed address position of `classes[0]`.
    start: usize,
    /// The original text, for diagnostics.
    source: String,
}

impl SuffixPattern {
    /// Recognise a suffix. The final `len()` address characters must match
    /// the pattern, with atom `i` constraining address position
    /// `ADDRESS_CHARS - len() + i`.
    ///
    /// # Errors
    ///
    /// [`SuffixError::Parse`] for grammar violations, and the structural
    /// variants when the demanded ending cannot occur in any onion address.
    pub fn parse(source: &str) -> Result<Self, SuffixError> {
        let pattern = Pattern::parse(source)?;
        let atoms = pattern.atoms();
        let start = ADDRESS_CHARS - atoms.len();
        let mut classes = Vec::with_capacity(atoms.len());
        for (i, atom) in atoms.iter().enumerate() {
            let class = atom.class();
            let position = start + i;
            match position {
                // 'd' is the fourth alphabet character, value 3; the terminal
                // position is fixed there by the version byte. A wildcard
                // admits it (and costs nothing); a set containing 'd' plus
                // more is ambiguous, and a set without it is unsatisfiable.
                55 if !class.is_any() => {
                    const D: u8 = 3;
                    if !(class.len() == 1 && class.admits_value(D)) {
                        let found = if class.admits_value(D) {
                            '?' // 'd' and dead weight; the others are a mistake
                        } else {
                            class.as_single().map_or('?', |c| char::from(c.as_ascii()))
                        };
                        return Err(SuffixError::NotTerminalCharacter { found });
                    }
                }
                // Only values 0, 8, 16, 24 survive the three fixed zero bits
                // of the version byte: a, i, q, y. A wildcard constrains
                // nothing and so lists none of them; only an explicit class
                // can ask for an impossible character.
                54 if !class.is_any() => {
                    for v in 0..32u8 {
                        if class.admits_value(v) && v % 8 != 0 {
                            let found =
                                Base32Char::from_value(v).map_or('?', |c| char::from(c.as_ascii()));
                            return Err(SuffixError::ImpossibleChecksumCharacter {
                                position,
                                found,
                            });
                        }
                    }
                }
                _ => {}
            }
            classes.push(class);
        }
        Ok(Self {
            classes,
            start,
            source: source.to_owned(),
        })
    }

    /// Whether this suffix matches an address body (the 56 characters before
    /// `.onion`). Body characters outside the alphabet never match — but the
    /// body comes from [`crate::address::OnionAddress`], which guarantees a
    /// valid alphabet throughout.
    #[must_use]
    pub fn matches_body(&self, body: &str) -> bool {
        let bytes = body.as_bytes();
        self.classes.iter().enumerate().all(|(i, class)| {
            bytes
                .get(self.start + i)
                .and_then(|&b| Base32Char::from_ascii(b))
                .is_some_and(|c| class.admits_value(c.value()))
        })
    }

    /// Base-2 log of the expected extra trials this suffix demands, given its
    /// position structure: position `p` can hold 32 values up to 53, four at
    /// 54, one at 55; a class admitting `n` of them costs the difference.
    #[must_use]
    pub fn difficulty_log2(&self) -> f64 {
        self.classes
            .iter()
            .enumerate()
            .map(|(i, class)| {
                if class.is_any() {
                    return 0.0; // "don't care", whatever the position allows
                }
                let universe: f64 = match self.start + i {
                    55 => 1.0,
                    54 => 4.0,
                    _ => 32.0,
                };
                universe.log2() - (class.len() as f64).log2()
            })
            .sum()
    }

    /// The original text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// A set of alternative suffixes: a candidate wins if it matches the prefix
/// set *and* any one suffix here. An empty set constrains nothing.
#[derive(Clone, Debug, Default)]
pub struct SuffixSet {
    patterns: Vec<SuffixPattern>,
}

impl SuffixSet {
    /// Build from alternatives. Duplicates are kept for reporting and priced
    /// as independent events when estimating difficulty, which is what
    /// [`super::compile::PatternSet::difficulty_log2`] does for prefixes too.
    #[must_use]
    pub fn new(patterns: Vec<SuffixPattern>) -> Self {
        Self { patterns }
    }

    /// Whether no suffix was requested — the overwhelmingly common case, and
    /// the one the search loop specialises on.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// The alternatives, for reporting.
    #[must_use]
    pub fn patterns(&self) -> &[SuffixPattern] {
        &self.patterns
    }

    /// Whether the address body satisfies at least one suffix. True when the
    /// set is empty: no suffix requested is no constraint imposed.
    #[must_use]
    pub fn matches_any(&self, body: &str) -> bool {
        self.patterns.is_empty() || self.patterns.iter().any(|p| p.matches_body(body))
    }

    /// Base-2 log of the expected extra trials the suffix demands. Same
    /// independent-events model as the prefix set: the cheapest alternative
    /// minus `log2(count)`.
    #[must_use]
    pub fn difficulty_log2(&self) -> f64 {
        if self.patterns.is_empty() {
            return 0.0;
        }
        let per = self
            .patterns
            .iter()
            .map(SuffixPattern::difficulty_log2)
            .fold(f64::INFINITY, f64::min);
        per - f64::from(self.patterns.len() as u32).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_iad_costs_seven_bits() {
        // 'i' at position 53 is uniform checksum (5 bits), 'a' at 54 is one
        // of four (2 bits), 'd' at 55 is free.
        let s = SuffixPattern::parse("iad").expect("valid");
        assert!((s.difficulty_log2() - 7.0).abs() < 1e-9);
        assert_eq!(s.source(), "iad");
    }

    #[test]
    fn the_terminal_d_is_free_and_everything_else_rejected() {
        assert!((SuffixPattern::parse("d").unwrap().difficulty_log2()).abs() < 1e-9);
        assert!(matches!(
            SuffixPattern::parse("x"),
            Err(SuffixError::NotTerminalCharacter { found: 'x' })
        ));
        // A wildcard at the terminal position admits 'd', so it is legal and
        // free; a class mixing 'd' with an impossible character is rejected.
        assert!((SuffixPattern::parse("?").unwrap().difficulty_log2()).abs() < 1e-9);
        assert!(matches!(
            SuffixPattern::parse("[dx]"),
            Err(SuffixError::NotTerminalCharacter { .. })
        ));
    }

    #[test]
    fn position_54_admits_only_a_i_q_y() {
        assert!((SuffixPattern::parse("ad").unwrap().difficulty_log2() - 2.0).abs() < 1e-9);
        assert!((SuffixPattern::parse("[ai]d").unwrap().difficulty_log2() - 1.0).abs() < 1e-9);
        // 'b' (value 1) cannot occur at position 54.
        assert!(matches!(
            SuffixPattern::parse("bd"),
            Err(SuffixError::ImpossibleChecksumCharacter {
                position: 54,
                found: 'b'
            })
        ));
        for c in ["ad", "id", "qd", "yd"] {
            assert!(SuffixPattern::parse(c).is_ok(), "{c} should be valid");
        }
    }

    #[test]
    fn grammar_still_applies() {
        // "lOck" fails the grammar (uppercase) before any structural rule.
        assert!(matches!(
            SuffixPattern::parse("lOck"),
            Err(SuffixError::Parse(_))
        ));
        // "lock" is well-formed grammar but unsatisfiable structure: 'c'
        // cannot sit at position 54. This is the flagship ask that physics
        // refuses, and the error says exactly why.
        assert!(matches!(
            SuffixPattern::parse("lock"),
            Err(SuffixError::ImpossibleChecksumCharacter {
                position: 54,
                found: 'c'
            })
        ));
    }

    #[test]
    fn four_character_suffix_prices_correctly() {
        // Positions 52..=55: 'z' twice at uniform checksum (5 bits each),
        // 'a' at 54 (2 bits), 'd' at 55 (0). Twelve bits, not twenty: the
        // version byte makes the tail of every address cheaper than users
        // assume.
        let s = SuffixPattern::parse("zzad").expect("valid");
        assert!((s.difficulty_log2() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn matches_the_terminal_of_a_real_address() {
        // DuckDuckGo's address, ending ...zad; from pattern_semantics tests.
        let body = "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad";
        assert_eq!(body.len(), ADDRESS_CHARS);
        let yes = SuffixPattern::parse("zad").unwrap();
        let no = SuffixPattern::parse("qad").unwrap();
        assert!(yes.matches_body(body));
        assert!(!no.matches_body(body));
        // Wildcard suffix: 'z?d' at 53..=55 must match, 'z?q' must not.
        assert!(SuffixPattern::parse("z?d").unwrap().matches_body(body));
        assert!(
            SuffixPattern::parse("z?q").is_err(),
            "q cannot end an address"
        );
    }

    #[test]
    fn set_semantics_or_across_alternatives() {
        let a = SuffixPattern::parse("zad").unwrap();
        let b = SuffixPattern::parse("qid").unwrap();
        let set = SuffixSet::new(vec![a, b]);
        assert_eq!(set.patterns().len(), 2);
        let body = "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad";
        assert!(set.matches_any(body));
        // Each is 5 + 2 = 7 bits; two alternatives halve the expected extra
        // work: 7 - 1 = 6.
        assert!((set.difficulty_log2() - 6.0).abs() < 1e-9);
        // The empty set constrains nothing.
        let none = SuffixSet::default();
        assert!(none.is_empty());
        assert!(none.matches_any(body));
        assert_eq!(none.difficulty_log2(), 0.0);
    }
}
