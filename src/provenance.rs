//! Provenance Guard and PFC (Prefrontal Cortex) Self-Citation Filter
//! Prevents LLM self-hallucination loops and enforces origin tracking.

use alloc::string::{String, ToString};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    UserUtterance,
    LlmGeneration,
    LlmSelfDescription,
    ExternalDoc,
    VerifiedExternal,
    UnknownLegacy,
}

impl SourceType {
    /// Check if node represents bot self-description requiring quarantine
    pub fn is_self_description(&self) -> bool {
        matches!(self, SourceType::LlmSelfDescription)
    }

    /// Default importance cap based on proven provenance
    pub fn max_importance_cap(&self) -> f32 {
        match self {
            SourceType::UserUtterance => 1.0,
            SourceType::VerifiedExternal => 1.0,
            SourceType::ExternalDoc => 0.85,
            SourceType::LlmGeneration => 0.70,
            SourceType::LlmSelfDescription => 0.40,
            SourceType::UnknownLegacy => 0.50,
        }
    }

    /// Score rank multiplier for retrieval
    pub fn rank_multiplier(&self) -> f32 {
        match self {
            SourceType::UserUtterance => 1.35,      // Primary user facts boosted
            SourceType::VerifiedExternal => 1.20,
            SourceType::ExternalDoc => 1.00,
            SourceType::LlmGeneration => 0.55,       // ungrounded model output — steeper discount
                                                     // (a scale benchmark showed 0.85 let the LLM
                                                     // background leak into top-k when relevant
                                                     // user facts were sparse)
            SourceType::LlmSelfDescription => 0.00,  // Quarantined from normal recall
            SourceType::UnknownLegacy => 0.90,
        }
    }
}

pub struct PfcGuard;

pub enum PfcAction {
    Accept,
    Elaborate(String),
    RejectDuplicate,
}

impl PfcGuard {
    /// PFC Self-Citation Check against existing canon/high-importance nodes
    /// Prevents saving repetitive or redundant facts to the memory graph.
    pub fn evaluate_similarity(similarity: f32, target_node_id: &str) -> PfcAction {
        if similarity >= 0.92 {
            PfcAction::RejectDuplicate
        } else if similarity >= 0.82 {
            PfcAction::Elaborate(target_node_id.to_string())
        } else {
            PfcAction::Accept
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_caps_and_quarantine() {
        assert_eq!(SourceType::UserUtterance.rank_multiplier(), 1.35);
        assert_eq!(SourceType::LlmGeneration.rank_multiplier(), 0.55);
        assert_eq!(SourceType::LlmSelfDescription.rank_multiplier(), 0.0);
        assert!(SourceType::LlmSelfDescription.is_self_description());
    }

    #[test]
    fn test_pfc_guard() {
        match PfcGuard::evaluate_similarity(0.95, "node_123") {
            PfcAction::RejectDuplicate => assert!(true),
            _ => panic!("Expected RejectDuplicate"),
        }
        match PfcGuard::evaluate_similarity(0.85, "node_123") {
            PfcAction::Elaborate(id) => assert_eq!(id, "node_123"),
            _ => panic!("Expected Elaborate"),
        }
        match PfcGuard::evaluate_similarity(0.50, "node_123") {
            PfcAction::Accept => assert!(true),
            _ => panic!("Expected Accept"),
        }
    }
}
