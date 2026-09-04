//! Immortal Canon mechanics (L1 / L2 Levels)
//! Canon nodes represent foundational truths (essence, roles, key user facts)
//! that are exempt from capacity eviction (`enforce_capacity`). The retrieval
//! multipliers are exposed as API but deliberately not used in recall ranking —
//! a benchmark showed importance boosts distort relevance (see c_api::astrum_memory_search).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CanonLevel {
    None = 0,
    L1Project = 1,      // Project-level canon (tracks, clips, studio role facts)
    L2Foundational = 2, // Core foundational canon (agent essence, primary user profile)
}

impl CanonLevel {
    pub fn is_canon(&self) -> bool {
        *self != CanonLevel::None
    }

    /// Exempt from capacity eviction and pruning
    pub fn is_immortal(&self) -> bool {
        self.is_canon()
    }

    /// Retrieval multiplier for ranking
    pub fn retrieval_multiplier(&self) -> f32 {
        match self {
            CanonLevel::None => 1.0,
            CanonLevel::L1Project => 1.15,
            CanonLevel::L2Foundational => 1.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canon_immortality_and_multipliers() {
        assert!(!CanonLevel::None.is_immortal());
        assert!(CanonLevel::L1Project.is_immortal());
        assert!(CanonLevel::L2Foundational.is_immortal());

        assert_eq!(CanonLevel::None.retrieval_multiplier(), 1.0);
        assert_eq!(CanonLevel::L1Project.retrieval_multiplier(), 1.15);
        assert_eq!(CanonLevel::L2Foundational.retrieval_multiplier(), 1.25);
    }
}
