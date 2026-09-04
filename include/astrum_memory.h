/*
 * Astrum HSAM — C-ABI header.
 *
 * Hypergraph memory engine for edge AI: provenance-weighted recall, canon retention
 * under memory pressure, optional int8 vector index. `no_std` + alloc; the consumer
 * supplies the allocator (see INTEGRATION.md).
 *
 * Build the library this header describes:
 *   cargo rustc --release --features std --crate-type staticlib          # host, with file persistence
 *   cargo rustc --release --no-default-features --features runtime \     # libc host, no_std
 *         --crate-type staticlib
 *   cargo rustc --release --no-default-features --features runtime,capi-int8 \
 *         --target thumbv7em-none-eabihf --crate-type staticlib          # Cortex-M4F, int8 index
 *
 * Memory ownership: every `char *` returned by this API is owned by the caller and MUST be
 * released with astrum_memory_free_string(). Never free() it, never keep it after destroy().
 *
 * Thread safety: one handle is internally serialized by a spinlock, so calls from multiple
 * threads are safe. It is NOT interrupt-safe — do not call from an ISR on bare metal.
 *
 * Version: 0.1.0
 */

#ifndef ASTRUM_MEMORY_H
#define ASTRUM_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. Create with astrum_memory_create(), release with astrum_memory_destroy(). */
typedef struct AstrumMemoryHandle AstrumMemoryHandle;

/* Provenance of a fact — the axis that keeps a model from citing its own output as truth.
 * ASTRUM_SOURCE_LLM_SELF_DESCRIPTION is hard-quarantined: such nodes are excluded from
 * recall entirely, not merely down-ranked (measured 0/600 in top-3, see RESULTS.md). */
typedef enum {
    ASTRUM_SOURCE_USER_UTTERANCE = 0,
    ASTRUM_SOURCE_LLM_GENERATION = 1,
    ASTRUM_SOURCE_LLM_SELF_DESCRIPTION = 2,
    ASTRUM_SOURCE_EXTERNAL_DOC = 3,
    ASTRUM_SOURCE_VERIFIED_EXTERNAL = 4,
    ASTRUM_SOURCE_UNKNOWN_LEGACY = 5
} AstrumSourceType;

/* Canon level — a canon node is never evicted under capacity pressure, however old or
 * rarely read it is. This is what preserves a safety rule an LRU policy would drop. */
typedef enum {
    ASTRUM_CANON_NONE = 0,
    ASTRUM_CANON_L1_PROJECT = 1,
    ASTRUM_CANON_L2_FOUNDATIONAL = 2
} AstrumCanonLevel;

/* ── lifecycle ─────────────────────────────────────────────────────────────── */

/* Create an empty engine. Returns NULL only if allocation fails. */
AstrumMemoryHandle *astrum_memory_create(void);

/* Release an engine and everything it owns. Passing NULL is a no-op. */
void astrum_memory_destroy(AstrumMemoryHandle *handle);

/* Number of nodes currently stored. Returns 0 for a NULL handle. */
size_t astrum_memory_node_count(const AstrumMemoryHandle *handle);

/* Adjacency weight between two topology cells (24-cell state topology).
 * Returns 1.0 for a NULL handle. */
float astrum_topological_boost(const AstrumMemoryHandle *handle,
                               uint8_t cell_a,
                               uint8_t cell_b);

/* ── write path ────────────────────────────────────────────────────────────── */

/*
 * Store a fact and, optionally, its embedding.
 *
 *   content       UTF-8 text of the fact (must not be NULL).
 *   tags_csv      comma-separated tags, or NULL/"" for none.
 *   source_type   one of AstrumSourceType — the provenance guard reads this.
 *   cell_id       topology cell 0..23; use 2 if you have no topology model.
 *   canon_level   one of AstrumCanonLevel; L1/L2 make the node eviction-proof.
 *   embedding     pointer to embedding_len floats, or NULL to store the fact unindexed.
 *                 The FIRST embedding fixes the index dimension; later ones must match
 *                 or they are silently not indexed.
 *
 * Returns a newly allocated node-id string (free with astrum_memory_free_string),
 * or NULL if handle is NULL or the id could not be allocated.
 */
char *astrum_memory_add_node(AstrumMemoryHandle *handle,
                             const char *content,
                             const char *tags_csv,
                             uint8_t source_type,
                             uint8_t cell_id,
                             uint8_t canon_level,
                             const float *embedding,
                             size_t embedding_len);

/*
 * Attach an emotional disposition to an existing node: valence [-1,1], arousal [0,1],
 * intensity [0,1] (out-of-range values are clamped, not rejected).
 * Returns 0 on success, 1 if handle/node_id is NULL, 2 if the node is unknown.
 */
int32_t astrum_memory_set_affect(AstrumMemoryHandle *handle,
                                 const char *node_id,
                                 float valence,
                                 float arousal,
                                 float intensity);

/* ── human feedback (retention, never ranking) ─────────────────────────────── */

/*
 * Record a HUMAN verdict on a recall: helpful != 0 if this node was the right thing to
 * surface, 0 if it was wrong or irrelevant.
 *
 * Call it from a human confirmation ONLY. A verdict taken from your own agent loop ("the
 * model used it, so it was good") feeds the model's output back in as evidence — the exact
 * self-echo this engine exists to prevent.
 *
 * The verdict changes how long the fact survives memory pressure and NOTHING about search
 * order. Confirmations cannot lift a node past the ceiling its provenance allows, and a
 * rejection weighs more than a confirmation.
 *
 * Returns 0 on success, 1 on NULL argument, 2 if the node is unknown.
 */
int32_t astrum_memory_record_feedback(AstrumMemoryHandle *handle,
                                      const char *node_id,
                                      int32_t helpful);

/*
 * Make every node confirmed at least min_confirmations times eviction-proof at canon_level
 * (see AstrumCanonLevel; 0 does nothing). Separate from recording feedback on purpose:
 * making a fact immortal is a decision you take, not a side effect of praise. Model
 * self-description is never promoted, however many confirmations it collects.
 * Returns the number of nodes promoted.
 */
size_t astrum_memory_promote_confirmed(AstrumMemoryHandle *handle,
                                       uint32_t min_confirmations,
                                       uint8_t canon_level);

/*
 * Drop nodes until at most max_nodes remain, least valuable first: lowest importance (which
 * feedback moves), then least recently used, then least used. Canon nodes are exempt and are
 * kept even if that leaves more than max_nodes. Evicted embeddings are dropped from the
 * vector index too, so the call actually returns memory.
 * Returns the number of nodes evicted.
 */
size_t astrum_memory_enforce_capacity(AstrumMemoryHandle *handle, size_t max_nodes);

/* ── recall path ───────────────────────────────────────────────────────────── */

/*
 * Weighted recall. Ranking is cosine similarity x provenance weight: candidates below a
 * relevance floor are dropped and quarantined sources are excluded outright, so a model's
 * own self-description cannot fill top-k when nothing else matches.
 *
 *   query/query_len   the query embedding; query_len must equal the index dimension.
 *   query_cell        accepted for API stability, no longer skews the ranking.
 *   top_k             maximum results.
 *
 * Returns a newly allocated JSON array string (free with astrum_memory_free_string):
 *   [{"node_id":"..","content":"..","cell_id":N,"cosine":0.9876,"score":0.9876}, ...]
 * sorted by score descending, or "[]" when there is no index, no query, or no match.
 * Never returns NULL for a valid handle.
 */
char *astrum_memory_search(const AstrumMemoryHandle *handle,
                           const float *query,
                           size_t query_len,
                           uint8_t query_cell,
                           size_t top_k);

/* ── persistence (only in builds with the `std` feature) ───────────────────── */

/*
 * Write the whole engine (graph + topology + vector index) to a JSON file, atomically.
 * Returns 0 on success, 1 on NULL argument, 2 on serialization/IO failure.
 * Absent from `no_std` builds — there is no filesystem to write to.
 */
int32_t astrum_memory_save(const AstrumMemoryHandle *handle, const char *path);

/*
 * Load an engine from a snapshot; returns a new handle (release with
 * astrum_memory_destroy) or NULL on error. A snapshot written by a build with a
 * different vector backend (see capi-int8 in INTEGRATION.md) is rejected.
 */
AstrumMemoryHandle *astrum_memory_load(const char *path);

/* ── misc ──────────────────────────────────────────────────────────────────── */

/* Library version string (free with astrum_memory_free_string). */
char *astrum_memory_version(void);

/* Release any string returned by this API. Passing NULL is a no-op. */
void astrum_memory_free_string(char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ASTRUM_MEMORY_H */
