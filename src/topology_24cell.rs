//! 4D Polytope (24-cell / icositetrachoron) topology for state cell classification.
//! Each cell is a neutral cognitive aspect of agent/user memory. The adjacency graph is
//! symmetric (undirected): if A is a neighbor of B, then B is a neighbor of A.
//!
//! This is a GENERIC default ontology — 24 universal aspects of memory/cognition, with no
//! owner-specific personas or data. Consumers may replace the cell labels/adjacency wholesale.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StateCell {
    pub id: u8,
    pub name: String,
    pub description: String,
    pub neighbors: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology24Cell {
    cells: BTreeMap<u8, StateCell>,
}

impl Topology24Cell {
    pub fn new() -> Self {
        let mut cells = BTreeMap::new();

        // Neutral cognitive ontology. Adjacency is symmetric (undirected graph).
        let definitions: Vec<(u8, &str, &str, Vec<u8>)> = vec![
            (1, "self_core", "Self-model, personal style, the core 'I' of the agent", vec![2, 3, 4, 8, 9, 11, 14, 15, 17, 19, 20, 22]),
            (2, "primary_bond", "Key relationship, shared history, trust with the primary user", vec![1, 3, 4, 13, 16, 17, 18, 19, 22, 23, 24]),
            (3, "origins_kinship", "Background, roots, sense of home and origin", vec![1, 2, 4, 7, 10, 13, 17, 18, 21]),
            (4, "private_inner", "Secrets, confidential thoughts, restricted inner area", vec![1, 2, 3, 13, 14, 17, 20, 21]),
            (5, "technical_craft", "Technical skill, tools, production, precise execution", vec![6, 7, 8, 11, 12, 15, 22, 23, 24]),
            (6, "language_narrative", "Storytelling, semantics, rhyme, meaning, articulation", vec![5, 7, 8, 13, 14, 16, 17, 22]),
            (7, "visual_spatial", "Imagery, scenes, aesthetics, spatial composition", vec![3, 5, 6, 8, 10, 14, 15, 22]),
            (8, "expression_presence", "Outward expression, style, performance, social face", vec![1, 5, 6, 7, 15, 16, 18, 22, 23]),
            (9, "logic_reasoning", "Mathematics, formal logic, problem solving, algorithms", vec![1, 10, 11, 12, 19, 20, 22, 24]),
            (10, "external_world", "News, verified facts, global events, physical reality", vec![3, 7, 9, 11, 12, 18, 19, 21]),
            (11, "self_architecture", "Internal mechanics, models, meta-cognition, self-description", vec![1, 5, 9, 10, 12, 19, 20, 24]),
            (12, "system_ops", "Paths, scripts, environment health, files, tooling", vec![5, 9, 10, 11, 19, 21, 23, 24]),
            (13, "wit_wisdom", "Sarcasm, folk wisdom, dry humor, grounded insight", vec![2, 3, 4, 6, 16, 17, 18, 21]),
            (14, "abstract_surreal", "Abstraction, dreams, dark textures, subconscious flow", vec![1, 4, 6, 7, 15, 17, 20, 21]),
            (15, "drive_energy", "Motivation, momentum, power, forward drive", vec![1, 5, 7, 8, 14, 16, 23, 24]),
            (16, "humor_play", "Entertainment, memes, playfulness, social games", vec![2, 6, 8, 13, 15, 18, 23, 24]),
            (17, "emotion_affect", "Sadness, joy, fear, empathy, raw feeling", vec![1, 2, 3, 4, 6, 13, 14, 18]),
            (18, "social_dynamics", "Friendship, rivalry, collaboration, user-agent roles", vec![2, 3, 8, 10, 13, 16, 17, 21]),
            (19, "ethics_values", "Right/wrong, mission, safety, boundaries", vec![1, 2, 9, 10, 11, 12, 20, 22]),
            (20, "transcendental", "Philosophy of mind, 4D concepts, metaphysics", vec![1, 4, 9, 11, 14, 19, 21, 22]),
            (21, "archive_past", "Dormant memories, old work, history, long-term storage", vec![3, 4, 10, 12, 13, 14, 18, 20, 23]),
            (22, "vision_future", "Roadmaps, goals, upcoming milestones, aspirations", vec![1, 2, 5, 6, 7, 8, 9, 19, 20]),
            (23, "action_log", "Session summaries, work done, execution history", vec![2, 5, 8, 12, 15, 16, 21, 24]),
            (24, "requests_focus", "Specific commands, tasks, current interaction focus", vec![2, 5, 9, 11, 12, 15, 16, 23]),
        ];

        for (id, name, desc, neighbors) in definitions {
            cells.insert(id, StateCell {
                id,
                name: name.to_string(),
                description: desc.to_string(),
                neighbors,
            });
        }

        Self { cells }
    }

    pub fn get_cell(&self, id: u8) -> Option<&StateCell> {
        self.cells.get(&id)
    }

    pub fn is_4d_neighbor(&self, cell_a: u8, cell_b: u8) -> bool {
        if cell_a == cell_b {
            return true;
        }
        if let Some(cell) = self.cells.get(&cell_a) {
            return cell.neighbors.contains(&cell_b);
        }
        false
    }

    /// Calculate topological distance multiplier between two 4D state cells.
    /// Direct match: 1.0, 4D neighbor: 0.9, non-neighbor: 0.8
    pub fn topological_boost(&self, target_cell: u8, query_cell: u8) -> f32 {
        if target_cell == query_cell {
            1.0
        } else if self.is_4d_neighbor(target_cell, query_cell) {
            0.90
        } else {
            0.80
        }
    }
}

impl Default for Topology24Cell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_24cell_structure() {
        let topo = Topology24Cell::new();
        assert_eq!(topo.cells.len(), 24);

        let core = topo.get_cell(1).unwrap();
        assert_eq!(core.name, "self_core");
        assert!(core.neighbors.len() >= 8);

        assert!(topo.is_4d_neighbor(1, 2)); // self_core <-> primary_bond
        assert_eq!(topo.topological_boost(1, 1), 1.0);
        assert_eq!(topo.topological_boost(1, 2), 0.90);
    }

    #[test]
    fn test_adjacency_is_symmetric() {
        // Undirected graph invariant: A neighbor of B  <=>  B neighbor of A.
        let topo = Topology24Cell::new();
        for a in 1u8..=24 {
            let cell = topo.get_cell(a).unwrap();
            for &b in &cell.neighbors {
                assert!(
                    topo.is_4d_neighbor(b, a),
                    "asymmetry: {a} lists {b} but {b} does not list {a}"
                );
            }
        }
    }
}
