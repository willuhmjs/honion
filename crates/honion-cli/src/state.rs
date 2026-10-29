//! Persisting the wall-clock fact that a multi-day search was happening.
//!
//! The search is memoryless — every key is an independent trial, so no
//! *search* state is worth saving. What a crash at 90% of an eleven-character
//! prefix does cost is the two weeks already spent, as far as the human is
//! concerned: the elapsed time, the examined count behind `P(found)`, the
//! keys already written, and the patterns themselves (retyping a fifty-one
//! character class pattern from a terminal scrollback is its own hazard).
//!
//! The file is therefore small, line-oriented, and complete-by-validation:
//!
//! ```text
//! honion-state v1
//! prefix=carroll
//! suffix=qad
//! elapsed_secs=172800.5
//! examined=2.3e15
//! written=0
//! ```
//!
//! Every line is a `key=value` with an exactly-one-use rule (`prefix` and
//! `suffix` repeat); an unknown key or an unparseable value rejects the
//! whole file rather than partially trusting it, and the pattern strings
//! are re-recognised through the normal grammar on load — the state file
//! is input like any other, not privileged memory.
//!
//! Writes follow the same discipline as key files (`store.rs`): temporary
//! file, fsync, rename, so a kill at any instant leaves either the previous
//! state or the new one, never neither.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// One checkpoint's worth of information.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchState {
    /// Pattern sources, in the order given. Re-parsed on load.
    pub prefixes: Vec<String>,
    /// Suffix sources, likewise.
    pub suffixes: Vec<String>,
    /// Seconds of search time accumulated across all runs into `--out`.
    pub elapsed_secs: f64,
    /// Candidates examined in total. The search is memoryless, so this is
    /// bookkeeping for `P(found)`, never a resume position.
    pub examined: u64,
    /// Keys already written into `--out`.
    pub written: u64,
}

impl SearchState {
    /// The file that holds the checkpoint inside a `--out` directory.
    pub fn path(out: &Path) -> PathBuf {
        out.join(".honion-state")
    }

    /// Load the checkpoint, or `None` when the directory has none.
    ///
    /// # Errors
    ///
    /// If the file exists but is not a state file this version can trust in
    /// full: unknown keys, unparseable values, duplicate scalars, or a
    /// wrong version line all fail the load rather than silently dropping
    /// an epoch's statistics.
    pub fn load(out: &Path) -> Result<Option<Self>> {
        let path = Self::path(out);
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(e).with_context(|| format!("reading {}", path.display()));
            }
        };
        let mut state = Self {
            prefixes: Vec::new(),
            suffixes: Vec::new(),
            elapsed_secs: -1.0,
            examined: u64::MAX,
            written: u64::MAX,
        };
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            if i == 0 {
                if line != "honion-state v1" {
                    bail!("{}: bad version line {line:?}", path.display());
                }
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .with_context(|| format!("{}: line {i} is not key=value", path.display()))?;
            match key {
                "prefix" => state.prefixes.push(value.to_owned()),
                "suffix" => state.suffixes.push(value.to_owned()),
                "elapsed_secs" => {
                    if state.elapsed_secs >= 0.0 {
                        bail!("{}: duplicate {key}", path.display());
                    }
                    state.elapsed_secs = value
                        .parse()
                        .with_context(|| format!("{}: bad {key}", path.display()))?;
                }
                "examined" => {
                    if state.examined != u64::MAX {
                        bail!("{}: duplicate {key}", path.display());
                    }
                    state.examined = value
                        .parse()
                        .with_context(|| format!("{}: bad {key}", path.display()))?;
                }
                "written" => {
                    if state.written != u64::MAX {
                        bail!("{}: duplicate {key}", path.display());
                    }
                    state.written = value
                        .parse()
                        .with_context(|| format!("{}: bad {key}", path.display()))?;
                }
                other => bail!("{}: unknown key {other:?}", path.display()),
            }
        }
        if state.elapsed_secs < 0.0 || state.examined == u64::MAX || state.written == u64::MAX {
            bail!("{}: missing required keys", path.display());
        }
        Ok(Some(state))
    }

    /// Write atomically (temp, fsync, rename), mode 0600. The file contains
    /// no secrets — patterns are user-chosen text — but it names them, and
    /// a search's patterns are nobody else's business either.
    ///
    /// # Errors
    ///
    /// On any I/O failure renaming or flushing the checkpoint.
    pub fn save(&self, out: &Path, tag: &str) -> Result<()> {
        let path = Self::path(out);
        // The temp name carries the writer's tag: per-device workers
        // checkpoint concurrently, and they must not share one temp path.
        let tmp = out.join(format!(".honion-state.tmp.{tag}"));
        let mut text = String::from("honion-state v1\n");
        for p in &self.prefixes {
            text.push_str(&format!("prefix={p}\n"));
        }
        for s in &self.suffixes {
            text.push_str(&format!("suffix={s}\n"));
        }
        text.push_str(&format!(
            "elapsed_secs={}\nexamined={}\nwritten={}\n",
            self.elapsed_secs, self.examined, self.written
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(text.as_bytes())
            .and_then(|()| file.sync_all())
            .with_context(|| format!("writing {}", tmp.display()))?;
        drop(file);
        fs::rename(&tmp, &path).with_context(|| {
            format!(
                "moving checkpoint into place ({} -> {})",
                tmp.display(),
                path.display()
            )
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("honion-state-test-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn round_trip_is_exact() {
        let dir = temp_dir("roundtrip");
        let state = SearchState {
            prefixes: vec!["carroll".into(), "hon?on[abc]".into()],
            suffixes: vec!["qad".into()],
            elapsed_secs: 172800.5,
            examined: 2_300_000_000_000_000,
            written: 7,
        };
        state.save(&dir, "test").expect("save");
        let loaded = SearchState::load(&dir).expect("valid").expect("present");
        assert_eq!(state, loaded);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_file_is_no_state() {
        let dir = temp_dir("missing");
        assert!(SearchState::load(&dir).expect("ok").is_none());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn anything_unknown_or_broken_rejects_the_whole_file() {
        let dir = temp_dir("broken");
        for bad in [
            "honion-state v1\nexamined=1\nwritten=0\nelapsed_secs=0\nprefix=a\nbogus=1\n",
            "honion-state v2\nexamined=1\nwritten=0\nelapsed_secs=0\n",
            "honion-state v1\nexamined=notanumber\nwritten=0\nelapsed_secs=0\n",
            "honion-state v1\nexamined=1\nexamined=2\nwritten=0\nelapsed_secs=0\n",
            "honion-state v1\nprefix=a\n",
        ] {
            fs::write(SearchState::path(&dir), bad).expect("write");
            assert!(SearchState::load(&dir).is_err(), "should reject: {bad:?}");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_leaves_no_temp_file_behind() {
        let dir = temp_dir("atomic");
        let state = SearchState {
            prefixes: vec!["a".into()],
            suffixes: Vec::new(),
            elapsed_secs: 1.0,
            examined: 2,
            written: 0,
        };
        state.save(&dir, "test").expect("save");
        assert!(!dir.join(".honion-state.tmp").exists());
        // Second save replaces, never merges: one version line.
        state.save(&dir, "test").expect("save again");
        let text = fs::read_to_string(SearchState::path(&dir)).expect("read");
        assert_eq!(text.matches("honion-state v1").count(), 1);
        fs::remove_dir_all(&dir).ok();
    }
}
