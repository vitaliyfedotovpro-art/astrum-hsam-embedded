//! Persistence — save/load the full engine state as JSON.
//!
//! The engine is in-memory by default; this module serializes the graph nexus and the
//! vector index to a single JSON file so an engine survives a restart. The topology is
//! part of the nexus (it is data once labels/adjacency may be customized), so it round-trips
//! too. Atomic write: serialize to a temp file, then rename over the target.

use crate::graph_nexus::MemoryGraphNexus;
use crate::vector_index::{CapiIndex, CAPI_INDEX_KIND};

use alloc::string::{String, ToString};
use serde::{Deserialize, Serialize};

/// Format version, so future changes can migrate old snapshots deliberately.
const SNAPSHOT_VERSION: u32 = 1;

/// Snapshots written before `index_kind` existed were always f32 — that is the only
/// backend the C-ABI had at the time, so defaulting is a fact, not a guess.
fn default_index_kind() -> String {
    "f32".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub version: u32,
    pub nexus: MemoryGraphNexus,
    pub index: Option<CapiIndex>,
    /// Which vector backend wrote this file — `"f32"` or `"int8"`. The two encodings are
    /// not interchangeable, so loading a foreign one must fail loudly rather than
    /// deserialize into nonsense.
    #[serde(default = "default_index_kind")]
    pub index_kind: String,
}

impl Snapshot {
    pub fn new(nexus: MemoryGraphNexus, index: Option<CapiIndex>) -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            nexus,
            index,
            index_kind: CAPI_INDEX_KIND.to_string(),
        }
    }

    /// Reject a snapshot written by the other build before its contents are trusted.
    /// A file with no index at all is portable — there is nothing encoding-specific in it.
    pub fn check_index_kind(&self) -> Result<(), String> {
        if self.index.is_none() || self.index_kind == CAPI_INDEX_KIND {
            return Ok(());
        }
        Err(alloc::format!(
            "snapshot vector index is '{}', this build is '{}' — rebuild with the matching \
             capi-int8 feature or re-embed the corpus",
            self.index_kind,
            CAPI_INDEX_KIND
        ))
    }

    /// Serialize to JSON bytes (always available, no_std-safe).
    pub fn to_json_vec(&self) -> Result<alloc::vec::Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize from JSON bytes (always available, no_std-safe).
    pub fn from_json_slice(data: &[u8]) -> Result<Snapshot, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// Deserialize, but turn the "written by the other build" case into a message that
    /// says so. A foreign snapshot fails inside serde first (f32 stores `[id, vec]`,
    /// int8 stores `[id, codes, scale]`), which on its own reads as an opaque parse
    /// error — so on failure we re-read just the header to name the real cause.
    pub fn from_json_slice_checked(data: &[u8]) -> Result<Snapshot, String> {
        match Snapshot::from_json_slice(data) {
            Ok(snap) => {
                snap.check_index_kind()?;
                Ok(snap)
            }
            Err(e) => {
                if let Ok(head) = serde_json::from_slice::<SnapshotHeader>(data) {
                    if head.index_kind != CAPI_INDEX_KIND {
                        return Err(alloc::format!(
                            "snapshot vector index is '{}', this build is '{}' — rebuild with \
                             the matching capi-int8 feature or re-embed the corpus",
                            head.index_kind,
                            CAPI_INDEX_KIND
                        ));
                    }
                }
                Err(alloc::format!("{}", e))
            }
        }
    }
}

/// Just enough of a snapshot to identify who wrote it. Unknown fields (nexus, index)
/// are skipped without being validated, so this parses even when the full struct cannot.
#[derive(Debug, Deserialize)]
struct SnapshotHeader {
    #[serde(default = "default_index_kind")]
    index_kind: String,
}

#[cfg(feature = "std")]
impl Snapshot {
    /// Serialize to JSON and write atomically (temp file + rename).
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read and deserialize a snapshot from disk. A file written by the other index
    /// backend is rejected with a message naming both kinds, not an opaque parse error.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Snapshot> {
        let json = std::fs::read(path)?;
        Snapshot::from_json_slice_checked(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::CanonLevel;
    use crate::provenance::SourceType;

    #[test]
    fn test_snapshot_buffer_roundtrip() {
        let mut nexus = MemoryGraphNexus::new();
        let id = nexus.create_node(
            "user likes tea".to_string(),
            "pref".to_string(),
            vec!["preference".to_string()],
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::L2Foundational,
        );
        let mut index = CapiIndex::new(3);
        index.insert(id.clone(), vec![1.0, 0.0, 0.0]).unwrap();

        let snap = Snapshot::new(nexus, Some(index));
        let bytes = snap.to_json_vec().unwrap();
        let loaded = Snapshot::from_json_slice(&bytes).unwrap();

        assert_eq!(loaded.version, SNAPSHOT_VERSION);
        assert_eq!(loaded.nexus.len(), 1);
        assert!(loaded.nexus.get_node().contains_key(&id));
        assert_eq!(loaded.index.as_ref().unwrap().len(), 1);
        assert_eq!(loaded.nexus.topology().topological_boost(1, 2), 0.90);
    }

    #[test]
    fn test_snapshot_records_and_checks_index_kind() {
        // Whatever this build is, its own snapshot passes the check and says so.
        let snap = Snapshot::new(MemoryGraphNexus::new(), None);
        assert_eq!(snap.index_kind, CAPI_INDEX_KIND);
        let bytes = snap.to_json_vec().unwrap();
        assert!(Snapshot::from_json_slice_checked(&bytes).is_ok());

        // A file written by the OTHER build must fail with a message naming both kinds,
        // not with an opaque serde error. (Its index encoding never parses here, so the
        // header re-read is what produces the diagnosis.)
        let other = if CAPI_INDEX_KIND == "f32" {
            "int8"
        } else {
            "f32"
        };
        let foreign = alloc::format!(
            r#"{{"version":1,"nexus":null,"index":{{"dimension":3,"vectors":[]}},"index_kind":"{}"}}"#,
            other
        );
        let err = Snapshot::from_json_slice_checked(foreign.as_bytes()).unwrap_err();
        assert!(
            err.contains(other),
            "error should name the foreign kind: {err}"
        );
        assert!(
            err.contains(CAPI_INDEX_KIND),
            "error should name this build: {err}"
        );
    }

    #[test]
    fn test_legacy_snapshot_defaults_to_f32() {
        // Snapshots written before the field existed carry no `index_kind` — they were
        // always f32, and that is what the default must report.
        let head: SnapshotHeader = serde_json::from_slice(br#"{"version":1}"#).unwrap();
        assert_eq!(head.index_kind, "f32");
    }

    #[cfg(all(test, feature = "std"))]
    #[test]
    fn test_snapshot_roundtrip() {
        let mut nexus = MemoryGraphNexus::new();
        let id = nexus.create_node(
            "user likes tea".to_string(),
            "pref".to_string(),
            vec!["preference".to_string()],
            SourceType::UserUtterance,
            2,
            None,
            CanonLevel::L2Foundational,
        );
        let mut index = CapiIndex::new(3);
        index.insert(id.clone(), vec![1.0, 0.0, 0.0]).unwrap();

        let path = std::env::temp_dir().join("astrum_hsam_test_snapshot.json");
        let snap = Snapshot::new(nexus, Some(index));
        snap.save(&path).unwrap();

        let loaded = Snapshot::load(&path).unwrap();
        assert_eq!(loaded.version, SNAPSHOT_VERSION);
        assert_eq!(loaded.nexus.len(), 1);
        assert!(loaded.nexus.get_node().contains_key(&id));
        assert_eq!(loaded.index.as_ref().unwrap().len(), 1);
        // Topology round-trips and still answers adjacency.
        assert_eq!(loaded.nexus.topology().topological_boost(1, 2), 0.90);

        let _ = std::fs::remove_file(&path);
    }
}
