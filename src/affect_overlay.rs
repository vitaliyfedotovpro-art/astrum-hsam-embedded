//! Affect-Overlay Engine v1 — Memory-grounded emotional disposition layer
//! Calculates real-time Valence, Arousal, and Intensity from recalled memory nodes.

use alloc::string::String;
use alloc::string::ToString;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AffectState {
    pub valence: f32,   // [-1.0, 1.0] (Negative <-> Positive)
    pub arousal: f32,   // [0.0, 1.0]  (Calm <-> Excited)
    pub intensity: f32, // [0.0, 1.0]  (Emotional weight)
    pub source: String,
    pub timestamp: u64,
}

impl Default for AffectState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.2,
            intensity: 0.1,
            source: "neutral_baseline".to_string(),
            timestamp: 0,
        }
    }
}

pub struct AffectOverlayEngine {
    current_mood: AffectState,
    inertia_beta: f32, // e.g., 0.75 for smooth emotional drift
    tau_seconds: f32,  // emotional memory decay half-life in seconds (e.g. 3600.0)
}

impl AffectOverlayEngine {
    pub fn new(inertia_beta: f32, tau_seconds: f32) -> Self {
        Self {
            current_mood: AffectState::default(),
            inertia_beta,
            tau_seconds,
        }
    }

    pub fn current_mood(&self) -> &AffectState {
        &self.current_mood
    }

    /// Compute emergent mood readout from a set of activated/recalled memory nodes
    pub fn compute_mood(&mut self, recalled_nodes: &[(&AffectState, f32, u64)], now: u64) {
        if recalled_nodes.is_empty() {
            return;
        }

        let mut total_weight = 0.0f32;
        let mut weighted_v = 0.0f32;
        let mut weighted_a = 0.0f32;
        let mut weighted_i = 0.0f32;

        for (affect, relevance, node_time) in recalled_nodes {
            let delta_sec = (now.saturating_sub(*node_time)) as f32;
            let recency = libm::expf(-delta_sec / self.tau_seconds);
            let weight = recency * (0.3 + 0.4 * affect.intensity + 0.3 * relevance);

            weighted_v += affect.valence * weight;
            weighted_a += affect.arousal * weight;
            weighted_i += affect.intensity * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let raw_v = (weighted_v / total_weight).clamp(-1.0, 1.0);
            let raw_a = (weighted_a / total_weight).clamp(0.0, 1.0);
            let raw_i = (weighted_i / total_weight).clamp(0.0, 1.0);

            // Apply emotional inertia
            let b = self.inertia_beta;
            self.current_mood.valence = b * self.current_mood.valence + (1.0 - b) * raw_v;
            self.current_mood.arousal = b * self.current_mood.arousal + (1.0 - b) * raw_a;
            self.current_mood.intensity = b * self.current_mood.intensity + (1.0 - b) * raw_i;
            self.current_mood.timestamp = now;
        }
    }

    /// Calculate mood congruence rank nudge for a node
    /// Soft rank multiplier: 1.0 (neutral) -> max boost 1.15 when mood matches node affect
    pub fn mood_congruence_nudge(&self, node_affect: &AffectState) -> f32 {
        if self.current_mood.intensity < 0.15 {
            return 1.0; // Neutral mood = no rank distortion
        }

        let dv = node_affect.valence - self.current_mood.valence;
        let da = node_affect.arousal - self.current_mood.arousal;
        let dist_sq = dv * dv + da * da;

        let congruence = libm::expf(-dist_sq / 0.5);
        1.0 + 0.12 * congruence * self.current_mood.intensity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affect_overlay_inertia() {
        let mut engine = AffectOverlayEngine::new(0.75, 3600.0);
        let now: u64 = 1000;

        let positive_affect = AffectState {
            valence: 0.8,
            arousal: 0.6,
            intensity: 0.7,
            source: "user".to_string(),
            timestamp: now,
        };

        engine.compute_mood(&[(&positive_affect, 0.9, now)], now);
        let mood = engine.current_mood();

        // Mood should drift towards positive, moderated by inertia
        assert!(mood.valence > 0.0 && mood.valence < 0.8);
        assert!(mood.arousal > 0.2);
    }
}
