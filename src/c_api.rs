//! C-ABI Foreign Function Interface (FFI) exports.
//! Enables seamless linking from C++, Swift, Go, C#, or ROS2 nodes.
//! The handle is internally synchronized (a single SpinMutex around the engine state),
//! so calls on one handle may come from multiple threads concurrently.

use crate::affect_overlay::AffectState;
use crate::canon::CanonLevel;
use crate::graph_nexus::MemoryGraphNexus;
#[cfg(feature = "std")]
use crate::persistence::Snapshot;
use crate::provenance::SourceType;
use crate::vector_index::CapiIndex;

use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::ffi::{c_char, CStr};
use core::ptr;

// ── spinlock (replaces std::sync::Mutex for no_std) ─────────────────────────

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// Minimal spinlock. NOTE: not interrupt-safe on bare metal without a critical section;
/// intended for main-loop use. Replaces std::sync::Mutex for no_std.
pub struct SpinMutex<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for SpinMutex<T> {}
unsafe impl<T: Send> Send for SpinMutex<T> {}

impl<T> SpinMutex<T> {
    pub const fn new(v: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(v),
        }
    }

    pub fn lock(&self) -> SpinGuard<'_, T> {
        while self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        SpinGuard { m: self }
    }
}

pub struct SpinGuard<'a, T> {
    m: &'a SpinMutex<T>,
}

impl<'a, T> core::ops::Deref for SpinGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.m.data.get() }
    }
}

impl<'a, T> core::ops::DerefMut for SpinGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.m.data.get() }
    }
}

impl<'a, T> Drop for SpinGuard<'a, T> {
    fn drop(&mut self) {
        self.m.locked.store(false, Ordering::Release);
    }
}

// ── engine state ─────────────────────────────────────────────────────────────

struct EngineState {
    nexus: MemoryGraphNexus,
    // Lazy: dimension fixed by the first embedding added. `CapiIndex` is the exact f32
    // index by default, or the int8 one under `--features capi-int8` (see vector_index).
    index: Option<CapiIndex>,
}

pub struct AstrumMemoryHandle {
    state: SpinMutex<EngineState>,
}

impl AstrumMemoryHandle {
    fn lock(&self) -> SpinGuard<'_, EngineState> {
        self.state.lock()
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn source_from_u8(v: u8) -> SourceType {
    match v {
        0 => SourceType::UserUtterance,
        1 => SourceType::LlmGeneration,
        2 => SourceType::LlmSelfDescription,
        3 => SourceType::ExternalDoc,
        4 => SourceType::VerifiedExternal,
        _ => SourceType::UnknownLegacy,
    }
}

fn canon_from_u8(v: u8) -> CanonLevel {
    match v {
        1 => CanonLevel::L1Project,
        2 => CanonLevel::L2Foundational,
        _ => CanonLevel::None,
    }
}

// ── lifecycle ────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn astrum_memory_create() -> *mut AstrumMemoryHandle {
    let handle = Box::new(AstrumMemoryHandle {
        state: SpinMutex::new(EngineState {
            nexus: MemoryGraphNexus::new(),
            index: None,
        }),
    });
    Box::into_raw(handle)
}

/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently. The handle is
/// freed here and must not be used, or destroyed again, afterwards.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_destroy(handle: *mut AstrumMemoryHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_node_count(handle: *const AstrumMemoryHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    let s = h.lock();
    s.nexus.len()
}

/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
#[no_mangle]
pub unsafe extern "C" fn astrum_topological_boost(
    handle: *const AstrumMemoryHandle,
    cell_a: u8,
    cell_b: u8,
) -> f32 {
    if handle.is_null() {
        return 1.0;
    }
    let h = &*handle;
    let s = h.lock();
    s.nexus.topology().topological_boost(cell_a, cell_b)
}

// ── write path ───────────────────────────────────────────────────────────────

/// Add a node and (optionally) its embedding.
/// - `tags_csv`: comma-separated tags, may be NULL/empty.
/// - `source_type`: 0=user_utterance 1=llm_generation 2=llm_self_description
///   3=external_doc 4=verified_external 5=unknown_legacy
/// - `canon_level`: 0=none 1=L1_project 2=L2_foundational
/// - `embedding`/`embedding_len`: pass a non-null pointer to index the node for search.
///   The first embedding added fixes the index dimension; later ones must match.
///
/// Returns a heap node-id string (free with astrum_memory_free_string), or NULL on error.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
/// Every non-NULL string argument must be a NUL-terminated C string. If `embedding` is non-NULL it must point to at
/// least `embedding_len` readable `f32`s; a shorter buffer reads out of bounds. The returned
/// pointer is owned by the caller and must be released with `astrum_memory_free_string`.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_add_node(
    handle: *mut AstrumMemoryHandle,
    content: *const c_char,
    tags_csv: *const c_char,
    source_type: u8,
    cell_id: u8,
    canon_level: u8,
    embedding: *const f32,
    embedding_len: usize,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let h = &*handle;
    let mut s = h.lock();

    let content_s = cstr_to_string(content);
    let tags: Vec<String> = cstr_to_string(tags_csv)
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let node_id = s.nexus.create_node(
        content_s.clone(),
        content_s,
        tags,
        source_from_u8(source_type),
        cell_id,
        None,
        canon_from_u8(canon_level),
    );

    // Optional embedding → vector index (lazy dimension init on first insert).
    if !embedding.is_null() && embedding_len > 0 {
        let vec = unsafe { core::slice::from_raw_parts(embedding, embedding_len) }.to_vec();
        if s.index.is_none() {
            s.index = Some(CapiIndex::new(embedding_len));
        }
        if let Some(idx) = s.index.as_mut() {
            let _ = idx.insert(node_id.clone(), vec);
        }
    }

    match CString::new(node_id) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Set the emotional disposition (valence [-1,1], arousal [0,1], intensity [0,1]) of an
/// existing node — the write path that feeds the affect-overlay engine at recall time.
/// Returns 0 on success, non-zero if the handle/node_id is null or the node is unknown.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
/// Every non-NULL string argument must be a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_set_affect(
    handle: *mut AstrumMemoryHandle,
    node_id: *const c_char,
    valence: f32,
    arousal: f32,
    intensity: f32,
) -> i32 {
    if handle.is_null() || node_id.is_null() {
        return 1;
    }
    let h = &*handle;
    let mut s = h.lock();
    let id = cstr_to_string(node_id);
    let affect = AffectState {
        valence: valence.clamp(-1.0, 1.0),
        arousal: arousal.clamp(0.0, 1.0),
        intensity: intensity.clamp(0.0, 1.0),
        source: "c_api".to_string(),
        timestamp: 0,
    };
    if s.nexus.set_node_affect(&id, affect) {
        0
    } else {
        2
    }
}

/// Record a HUMAN verdict on a recall: `helpful != 0` means the node was the right thing to
/// surface, `0` means it was wrong or irrelevant.
///
/// Call this from a human confirmation only. A verdict derived from your own agent loop
/// ("the model used it, so it was good") feeds the model's output back in as evidence — the
/// self-echo this engine exists to prevent, wearing a different hat.
///
/// The verdict changes how long the fact survives memory pressure and NOTHING about search
/// order. Importance is clamped by provenance, so confirmations cannot raise model
/// self-description to the standing of the user's own words; rejection weighs more than
/// confirmation (-0.20 vs +0.15).
///
/// Returns 0 on success, 1 if the handle/node_id is NULL, 2 if the node is unknown.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
/// Every non-NULL string argument must be a NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_record_feedback(
    handle: *mut AstrumMemoryHandle,
    node_id: *const c_char,
    helpful: i32,
) -> i32 {
    if handle.is_null() || node_id.is_null() {
        return 1;
    }
    let h = &*handle;
    let mut s = h.lock();
    let id = cstr_to_string(node_id);
    if s.nexus.record_human_feedback(&id, helpful != 0) {
        0
    } else {
        2
    }
}

/// Make every node confirmed at least `min_confirmations` times eviction-proof at
/// `canon_level` (1 = L1 project, 2 = L2 foundational; 0 does nothing). Separate from
/// recording feedback on purpose: making a fact immortal is a decision, taken when you
/// choose, not a side effect of praise. Model self-description is never promoted.
/// Returns the number of nodes promoted.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_promote_confirmed(
    handle: *mut AstrumMemoryHandle,
    min_confirmations: u32,
    canon_level: u8,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    let mut s = h.lock();
    s.nexus
        .promote_confirmed_to_canon(min_confirmations, canon_from_u8(canon_level))
}

/// Drop nodes until at most `max_nodes` remain, evicting the least valuable first: lowest
/// importance (which human feedback moves), then least recently accessed, then least used.
/// Canon nodes are exempt and are kept even if that leaves more than `max_nodes`.
///
/// The evicted nodes' embeddings are dropped from the vector index too — otherwise the call
/// would free almost nothing, since the embedding is the bulk of a node's cost.
/// Returns the number of nodes evicted.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_enforce_capacity(
    handle: *mut AstrumMemoryHandle,
    max_nodes: usize,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    let mut s = h.lock();
    let evicted = s.nexus.enforce_capacity(max_nodes);
    if evicted > 0 {
        // Split the borrow: the surviving ids are read from the nexus, the index is pruned.
        let state = &mut *s;
        if let Some(idx) = state.index.as_mut() {
            let nodes = state.nexus.get_node();
            idx.retain_ids(|id| nodes.contains_key(id));
        }
    }
    evicted
}

// ── recall path ──────────────────────────────────────────────────────────────

/// Weighted associative recall. Ranks by cosine similarity re-weighted by
/// provenance (anti self-echo). Canon/topology multipliers are deliberately NOT
/// part of the rank — a recall@k benchmark showed they hurt relevance (see below).
/// `query_cell` is accepted for API stability but no longer skews the ranking.
/// Returns a heap JSON array string (free with astrum_memory_free_string):
///   [{"node_id","content","cell_id","cosine","score"}], sorted by score desc.
/// Returns "[]" if there is no index or no query.
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently. `query` must point to at
/// least `query_len` readable `f32`s. The returned pointer is owned by the caller and must be
/// released with `astrum_memory_free_string`.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_search(
    handle: *const AstrumMemoryHandle,
    query: *const f32,
    query_len: usize,
    query_cell: u8,
    top_k: usize,
) -> *mut c_char {
    let empty = || CString::new("[]").unwrap().into_raw();
    if handle.is_null() || query.is_null() || query_len == 0 || top_k == 0 {
        return empty();
    }
    let h = &*handle;
    let s = h.lock();
    let idx = match s.index.as_ref() {
        Some(i) => i,
        None => return empty(),
    };
    let q = unsafe { core::slice::from_raw_parts(query, query_len) };

    // Over-fetch on raw cosine, then re-rank with the provenance weight.
    let candidates = idx.search(q, top_k.saturating_mul(4).max(top_k));
    let nodes = s.nexus.get_node();

    // Ranking = cosine × provenance ONLY. A recall@k benchmark (examples/benchmark_recall)
    // showed that folding canon + topology multipliers into the rank HURTS relevance
    // (29% recall@1 vs 86% flat): an important-but-off-topic fact outranks the on-topic one.
    // Provenance-only demotes model self-echo without distorting real facts and BEATS flat
    // (100% vs 86% recall@1). Canon/topology belong to retention/decay, not to relevance rank.
    // `query_cell` is accepted for API stability but no longer skews the ranking.
    let _ = query_cell;
    // Relevance floor + quarantine EXCLUSION. A scale benchmark (examples/bench_scale) showed
    // that at thousands of nodes, ranking-only guarding leaks contamination: quarantined
    // self-echo (rank_multiplier 0) still fills top-k when nothing else clears the query, and
    // low-cosine background floods in. Fix: drop candidates below RELEVANCE_FLOOR cosine, and
    // drop quarantined sources outright rather than merely down-ranking them. Result: self-echo
    // contamination@3 goes 67% -> 0% at 5k nodes (vs the ranking-only version).
    const RELEVANCE_FLOOR: f32 = 0.25;
    let mut scored: Vec<(f32, f32, &crate::graph_nexus::MemoryNode)> = candidates
        .iter()
        .filter(|r| r.score >= RELEVANCE_FLOOR)
        .filter_map(|r| {
            nodes.get(&r.node_id).and_then(|n| {
                let w_prov = n.source_type.rank_multiplier();
                if w_prov <= 0.0 {
                    return None; // quarantined (self-description) — excluded, not down-ranked
                }
                Some((r.score * w_prov, r.score, n))
            })
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    scored.truncate(top_k);

    // Hand-built JSON (content escaped) — avoids a serde dependency on MemoryNode here.
    let mut out = String::from("[");
    for (i, (score, cosine, n)) in scored.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let esc = n.content.replace('\\', "\\\\").replace('"', "\\\"");
        out.push_str(&alloc::format!(
            "{{\"node_id\":\"{}\",\"content\":\"{}\",\"cell_id\":{},\"cosine\":{:.4},\"score\":{:.4}}}",
            n.id, esc, n.cell_id, cosine, score
        ));
    }
    out.push(']');

    match CString::new(out) {
        Ok(s) => s.into_raw(),
        Err(_) => empty(),
    }
}

// ── persistence ──────────────────────────────────────────────────────────────

/// Save the full engine state (graph + topology + vector index) to a JSON file.
/// Returns 0 on success, non-zero on error (null handle/path, or I/O failure).
///
/// # Safety
///
/// `handle` must be NULL or a live handle from `astrum_memory_create`/`astrum_memory_load`
/// that no other thread is destroying concurrently.
/// Every non-NULL string argument must be a NUL-terminated C string.
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_save(
    handle: *const AstrumMemoryHandle,
    path: *const c_char,
) -> i32 {
    if handle.is_null() || path.is_null() {
        return 1;
    }
    let h = &*handle;
    let s = h.lock();
    let path_s = cstr_to_string(path);
    // Clone the state into an owned snapshot (borrow-only view is enough here).
    let snap = Snapshot::new(s.nexus.clone(), s.index.clone());
    match snap.save(&path_s) {
        Ok(()) => 0,
        Err(_) => 2,
    }
}

/// Load an engine from a JSON snapshot. Returns a new handle (free with
/// astrum_memory_destroy), or NULL on error. The saved topology is restored.
///
/// # Safety
///
/// Every non-NULL string argument must be a NUL-terminated C string. The returned handle is owned by the
/// caller and must be released with `astrum_memory_destroy`.
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_load(path: *const c_char) -> *mut AstrumMemoryHandle {
    if path.is_null() {
        return ptr::null_mut();
    }
    let path_s = cstr_to_string(path);
    let snap = match Snapshot::load(&path_s) {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    let handle = Box::new(AstrumMemoryHandle {
        state: SpinMutex::new(EngineState {
            nexus: snap.nexus,
            index: snap.index,
        }),
    });
    Box::into_raw(handle)
}

// ── misc ─────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn astrum_memory_version() -> *mut c_char {
    let s = CString::new("Astrum HSAM Rust v0.1.0 C-ABI").unwrap();
    s.into_raw()
}

/// # Safety
///
/// `s` must be NULL or a pointer returned by this library's own string-returning
/// functions, passed exactly once. Freeing a foreign or already-freed pointer is undefined.
#[no_mangle]
pub unsafe extern "C" fn astrum_memory_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_api() {
        // Every call below crosses the C ABI, where the caller owns the
        // pointer contract; one block beats dotting `unsafe` through each line.
        unsafe {
            let handle = astrum_memory_create();
            assert!(!handle.is_null());

            let count = astrum_memory_node_count(handle);
            assert_eq!(count, 0);

            let boost = astrum_topological_boost(handle, 1, 2);
            assert_eq!(boost, 0.90);

            astrum_memory_destroy(handle);
        }
    }

    #[test]
    fn test_c_api_add_and_search() {
        // Every call below crosses the C ABI, where the caller owns the
        // pointer contract; one block beats dotting `unsafe` through each line.
        unsafe {
            let handle = astrum_memory_create();
            let mk = |s: &str| CString::new(s).unwrap();

            // Two user facts (cell 2) and one LLM self-note (cell 11), 3-dim embeddings.
            let e1 = [1.0f32, 0.0, 0.0];
            let e2 = [0.9f32, 0.1, 0.0];
            let e3 = [0.0f32, 0.0, 1.0];
            // Every returned id is freed: dropping one leaks it, which is exactly what Miri
            // caught here — the API hands ownership to the caller and has no way to take it back.
            let id1 = astrum_memory_add_node(
                handle,
                mk("user likes tea").as_ptr(),
                mk("pref").as_ptr(),
                0,
                2,
                0,
                e1.as_ptr(),
                3,
            );
            let id2 = astrum_memory_add_node(
                handle,
                mk("user likes coffee").as_ptr(),
                mk("pref").as_ptr(),
                0,
                2,
                0,
                e2.as_ptr(),
                3,
            );
            let id3 = astrum_memory_add_node(
                handle,
                mk("i am an engine").as_ptr(),
                mk("meta").as_ptr(),
                2,
                11,
                0,
                e3.as_ptr(),
                3,
            );
            astrum_memory_free_string(id1);
            astrum_memory_free_string(id2);
            astrum_memory_free_string(id3);

            assert_eq!(astrum_memory_node_count(handle), 3);

            // Query near e1: should return tea/coffee (user facts) ahead of the self-note.
            let q = [1.0f32, 0.0, 0.0];
            let json_ptr = astrum_memory_search(handle, q.as_ptr(), 3, 2, 3);
            let json = CStr::from_ptr(json_ptr).to_string_lossy().into_owned();
            astrum_memory_free_string(json_ptr);

            assert!(json.contains("tea"));
            assert!(json.starts_with('['));
            // Top result is a user fact, not the self-description.
            let first = json.split("},{").next().unwrap();
            assert!(first.contains("tea") || first.contains("coffee"));

            astrum_memory_destroy(handle);
        }
    }

    #[test]
    fn test_c_api_set_affect() {
        // Every call below crosses the C ABI, where the caller owns the
        // pointer contract; one block beats dotting `unsafe` through each line.
        unsafe {
            let handle = astrum_memory_create();
            let content = CString::new("stormy night conversation").unwrap();
            let id_ptr = astrum_memory_add_node(
                handle,
                content.as_ptr(),
                ptr::null(),
                0,
                17,
                0,
                ptr::null(),
                0,
            );
            assert!(!id_ptr.is_null());

            // Out-of-range values are clamped; unknown node id fails with 2.
            let rc = astrum_memory_set_affect(handle, id_ptr, -2.0, 0.9, 1.5);
            assert_eq!(rc, 0);
            let missing = CString::new("no-such-node").unwrap();
            assert_eq!(
                astrum_memory_set_affect(handle, missing.as_ptr(), 0.0, 0.0, 0.0),
                2
            );

            let h = &*handle;
            {
                let s = h.lock();
                let id = CStr::from_ptr(id_ptr).to_string_lossy().into_owned();
                let node = s.nexus.get_node().get(&id).unwrap();
                assert_eq!(node.affect.valence, -1.0); // clamped
                assert_eq!(node.affect.intensity, 1.0); // clamped
                assert_eq!(node.affect.source, "c_api");
            }

            astrum_memory_free_string(id_ptr);
            astrum_memory_destroy(handle);
        }
    }

    #[test]
    fn test_c_api_feedback_drives_eviction_and_frees_the_index() {
        // Every call below crosses the C ABI, where the caller owns the
        // pointer contract; one block beats dotting `unsafe` through each line.
        unsafe {
            let handle = astrum_memory_create();
            let mk = |s: &str| CString::new(s).unwrap();

            // Two equally valid user facts; the first is also the better match for the query.
            let e1 = [1.0f32, 0.0, 0.0];
            let e2 = [0.9f32, 0.1, 0.0];
            let id_bad = astrum_memory_add_node(
                handle,
                mk("misheard fact").as_ptr(),
                ptr::null(),
                0,
                2,
                0,
                e1.as_ptr(),
                3,
            );
            let id_good = astrum_memory_add_node(
                handle,
                mk("correct fact").as_ptr(),
                ptr::null(),
                0,
                2,
                0,
                e2.as_ptr(),
                3,
            );

            // A human says the close match was wrong and the other one was right.
            assert_eq!(astrum_memory_record_feedback(handle, id_bad, 0), 0);
            assert_eq!(astrum_memory_record_feedback(handle, id_good, 1), 0);
            assert_eq!(astrum_memory_record_feedback(handle, ptr::null(), 1), 1);
            let missing = CString::new("no-such-node").unwrap();
            assert_eq!(
                astrum_memory_record_feedback(handle, missing.as_ptr(), 1),
                2
            );

            // Under pressure the rejected fact goes, the confirmed one stays — despite being
            // the weaker cosine match.
            assert_eq!(astrum_memory_enforce_capacity(handle, 1), 1);
            assert_eq!(astrum_memory_node_count(handle), 1);
            {
                let h = &*handle;
                let s = h.lock();
                let good = CStr::from_ptr(id_good).to_string_lossy().into_owned();
                assert!(
                    s.nexus.get_node().contains_key(&good),
                    "confirmed fact should survive"
                );
                // The evicted node's embedding is gone too, or the eviction freed nothing.
                assert_eq!(
                    s.index.as_ref().unwrap().len(),
                    1,
                    "index must shed evicted vectors"
                );
            }

            // Promotion is explicit and makes the fact immortal.
            assert_eq!(astrum_memory_promote_confirmed(handle, 1, 2), 1);
            assert_eq!(astrum_memory_enforce_capacity(handle, 0), 0);
            assert_eq!(astrum_memory_node_count(handle), 1);

            astrum_memory_free_string(id_bad);
            astrum_memory_free_string(id_good);
            astrum_memory_destroy(handle);
        }
    }

    #[cfg(all(test, feature = "std"))]
    #[test]
    fn test_c_api_concurrent_access() {
        // Every call below crosses the C ABI, where the caller owns the
        // pointer contract; one block beats dotting `unsafe` through each line.
        unsafe {
            // The thread-safety claim in the header must hold in Rust terms too:
            // concurrent add_node + search on one handle, no data race (SpinMutex serializes).
            let handle = astrum_memory_create();

            // Carry the pointer itself across threads rather than an integer address. Casting
            // through usize erases provenance, and Miri then warns it can no longer check this
            // test for pointer bugs — which is the one thing the test exists to check.
            #[derive(Clone, Copy)]
            struct SharedHandle(*mut AstrumMemoryHandle);
            unsafe impl Send for SharedHandle {}
            let shared = SharedHandle(handle);

            let writers: Vec<_> = (0..4)
                .map(|t| {
                    std::thread::spawn(move || {
                        // Bind the whole struct: edition-2021 closures capture individual fields,
                        // and a bare `shared.0` would capture the raw pointer, which is not Send.
                        let shared = shared;
                        let h = shared.0;
                        for i in 0..50 {
                            let content = CString::new(alloc::format!("fact {t}-{i}")).unwrap();
                            let e = [i as f32, t as f32, 1.0];
                            let id = astrum_memory_add_node(
                                h,
                                content.as_ptr(),
                                ptr::null(),
                                0,
                                2,
                                0,
                                e.as_ptr(),
                                3,
                            );
                            astrum_memory_free_string(id);
                            let q = [1.0f32, 0.0, 0.0];
                            let json = astrum_memory_search(h, q.as_ptr(), 3, 0, 5);
                            astrum_memory_free_string(json);
                        }
                    })
                })
                .collect();
            for w in writers {
                w.join().unwrap();
            }

            assert_eq!(astrum_memory_node_count(handle), 200);
            astrum_memory_destroy(handle);
        }
    }
}
