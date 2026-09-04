//! Forman-Ricci Curvature Analysis for Hypergraphs
//! Measures topological curvature F(e) for edges and hyperedges to detect
//! information bottlenecks, community clusters, and prune noise (F < 0.10).

use alloc::string::String;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormanCurvatureResult {
    pub edge_id: String,
    pub curvature: f32,
    pub is_bottleneck: bool, // F < 0.0 indicates bottleneck/bridge between clusters
}

pub struct FormanRicciCalculator;

impl FormanRicciCalculator {
    /// Calculate Forman-Ricci curvature F(e) for an edge e = (u, v):
    /// F(e) = 4 - deg(u) - deg(v) + 3 * # triangles containing e
    pub fn calculate_edge_curvature(deg_u: usize, deg_v: usize, triangles_count: usize) -> f32 {
        4.0 - (deg_u as f32) - (deg_v as f32) + 3.0 * (triangles_count as f32)
    }

    /// Calculate Forman-Ricci curvature F(h) for a hyperedge h connecting node_ids:
    /// F(h) = |h| * (4 - sum(deg(v) for v in h) / |h|)
    pub fn calculate_hyperedge_curvature(node_degrees: &[usize]) -> f32 {
        let size = node_degrees.len();
        if size == 0 {
            return 0.0;
        }

        let sum_deg: usize = node_degrees.iter().sum();
        let avg_deg = (sum_deg as f32) / (size as f32);
        (size as f32) * (4.0 - avg_deg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forman_ricci_edge() {
        // Triangle edge (deg 2, deg 2, 1 triangle) -> F = 4 - 2 - 2 + 3 = 3 (dense cluster)
        let f_dense = FormanRicciCalculator::calculate_edge_curvature(2, 2, 1);
        assert_eq!(f_dense, 3.0);

        // Bridge edge (deg 5, deg 6, 0 triangles) -> F = 4 - 5 - 6 + 0 = -7 (bottleneck)
        let f_bridge = FormanRicciCalculator::calculate_edge_curvature(5, 6, 0);
        assert_eq!(f_bridge, -7.0);
        assert!(f_bridge < 0.0);
    }
}
