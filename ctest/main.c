/*
 * Astrum HSAM — C integration smoke test.
 *
 * Proves the header matches the shipped library: it links, the engine stores facts,
 * recall ranks a user fact above the model's own self-description, and a snapshot
 * round-trips. This is also the shortest working example for a consumer.
 *
 * Build (from the crate root):
 *   cargo rustc --release --features std --crate-type staticlib
 *   cc ctest/main.c -Iinclude -Ltarget/release -lastrum_memory -o ctest/ctest
 *   ./ctest/ctest
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "astrum_memory.h"

#define DIM 3

static int failures = 0;

static void check(int ok, const char *what) {
    printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

int main(void) {
    char *version = astrum_memory_version();
    printf("%s\n", version);
    astrum_memory_free_string(version);

    AstrumMemoryHandle *mem = astrum_memory_create();
    check(mem != NULL, "engine created");

    /* Two user facts and one model self-description, deliberately near-identical
     * in embedding space so that only provenance can separate them. */
    const float e_tea[DIM] = {1.0f, 0.0f, 0.0f};
    const float e_coffee[DIM] = {0.9f, 0.1f, 0.0f};
    const float e_self[DIM] = {0.98f, 0.02f, 0.0f};

    char *id_tea = astrum_memory_add_node(mem, "the user drinks tea in the morning", "habit",
                                          ASTRUM_SOURCE_USER_UTTERANCE, 2,
                                          ASTRUM_CANON_NONE, e_tea, DIM);
    check(id_tea != NULL, "user fact stored, id returned");

    char *id_coffee = astrum_memory_add_node(mem, "the user drinks coffee after lunch", "habit",
                                             ASTRUM_SOURCE_USER_UTTERANCE, 2,
                                             ASTRUM_CANON_NONE, e_coffee, DIM);

    char *id_self = astrum_memory_add_node(mem, "I remember that the user likes hot drinks",
                                           "meta", ASTRUM_SOURCE_LLM_SELF_DESCRIPTION, 11,
                                           ASTRUM_CANON_NONE, e_self, DIM);

    check(astrum_memory_node_count(mem) == 3, "three nodes counted");

    /* Affect is optional metadata on a stored node. */
    check(astrum_memory_set_affect(mem, id_tea, 0.6f, 0.2f, 0.5f) == 0, "affect set on a node");
    check(astrum_memory_set_affect(mem, "no-such-node", 0.0f, 0.0f, 0.0f) == 2,
          "unknown node id rejected");

    /* Recall: the self-description sits closest to the query, yet must not appear. */
    const float query[DIM] = {1.0f, 0.0f, 0.0f};
    char *json = astrum_memory_search(mem, query, DIM, 2, 3);
    printf("  recall: %s\n", json);
    check(strstr(json, "tea") != NULL, "user fact recalled");
    check(strstr(json, "I remember") == NULL, "model self-description quarantined from recall");
    astrum_memory_free_string(json);

    /* Human feedback: moves retention, leaves ranking alone. The coffee fact is the weaker
     * match for the query, but a human says it was the right recall and tea was not. */
    check(astrum_memory_record_feedback(mem, id_tea, 0) == 0, "rejection recorded");
    check(astrum_memory_record_feedback(mem, id_coffee, 1) == 0, "confirmation recorded");
    check(astrum_memory_record_feedback(mem, "no-such-node", 1) == 2, "unknown node rejected");

    char *ranked = astrum_memory_search(mem, query, DIM, 2, 3);
    const char *pos_tea = strstr(ranked, "tea");
    const char *pos_coffee = strstr(ranked, "coffee");
    check(pos_tea != NULL && pos_coffee != NULL && pos_tea < pos_coffee,
          "search order is unchanged by feedback");
    astrum_memory_free_string(ranked);

    /* Under pressure the rejected fact goes first, the confirmed one stays — even though it
     * is the WEAKER match for the query. Squeezing 3 nodes down to 1 drops the rejected fact
     * and the self-description, whose provenance caps its importance below a user fact. */
    check(astrum_memory_enforce_capacity(mem, 1) == 2, "pressure evicted the two least valued");
    char *survivor = astrum_memory_search(mem, query, DIM, 2, 3);
    printf("  after pressure: %s\n", survivor);
    check(strstr(survivor, "coffee") != NULL, "confirmed fact survived");
    check(strstr(survivor, "tea") == NULL, "rejected fact was evicted");
    astrum_memory_free_string(survivor);

    /* Promotion is a separate, deliberate act; afterwards the fact is immortal. */
    check(astrum_memory_promote_confirmed(mem, 1, ASTRUM_CANON_L1_PROJECT) == 1, "fact promoted to canon");
    check(astrum_memory_enforce_capacity(mem, 0) == 0, "canon survives total pressure");
    check(astrum_memory_node_count(mem) == 1, "one canon node remains");

    /* Snapshot round-trip through the filesystem. */
    const char *path = "ctest/roundtrip.json";
    check(astrum_memory_save(mem, path) == 0, "snapshot saved");
    AstrumMemoryHandle *restored = astrum_memory_load(path);
    check(restored != NULL, "snapshot loaded");
    check(restored != NULL && astrum_memory_node_count(restored) == 1, "node survived reload");

    char *json2 = astrum_memory_search(restored, query, DIM, 2, 3);
    check(strstr(json2, "coffee") != NULL, "recall works on the restored engine");
    astrum_memory_free_string(json2);
    remove(path);

    astrum_memory_free_string(id_tea);
    astrum_memory_free_string(id_coffee);
    astrum_memory_free_string(id_self);
    astrum_memory_destroy(restored);
    astrum_memory_destroy(mem);

    /* Abuse: every documented edge case, on a fresh engine. None of these may crash or
     * corrupt state — they are the paths a real integration hits on its bad days. */
    AstrumMemoryHandle *edge = astrum_memory_create();
    const float e3[DIM] = {0.0f, 1.0f, 0.0f};

    char *empty = astrum_memory_search(edge, query, DIM, 2, 3);
    check(strcmp(empty, "[]") == 0, "search on an empty engine returns []");
    astrum_memory_free_string(empty);

    char *no_embed = astrum_memory_add_node(edge, "fact without an embedding", NULL,
                                            ASTRUM_SOURCE_EXTERNAL_DOC, 2, ASTRUM_CANON_NONE,
                                            NULL, 0);
    check(no_embed != NULL, "node stored without an embedding");
    astrum_memory_free_string(no_embed);

    /* First embedding fixes the dimension; a mismatched one must not be indexed or crash. */
    char *dim_ok = astrum_memory_add_node(edge, "3-dim fact", NULL, 0, 2, 0, e3, DIM);
    const float wrong[2] = {1.0f, 0.0f};
    char *dim_bad = astrum_memory_add_node(edge, "2-dim fact", NULL, 0, 2, 0, wrong, 2);
    check(dim_ok != NULL && dim_bad != NULL, "dimension mismatch is survivable");
    astrum_memory_free_string(dim_ok);
    astrum_memory_free_string(dim_bad);

    char *k0 = astrum_memory_search(edge, query, DIM, 2, 0);
    check(strcmp(k0, "[]") == 0, "top_k = 0 returns []");
    astrum_memory_free_string(k0);

    char *qlen = astrum_memory_search(edge, query, 999, 2, 3);
    check(strcmp(qlen, "[]") == 0, "wrong query length returns []");
    astrum_memory_free_string(qlen);

    char *null_search = astrum_memory_search(NULL, query, DIM, 2, 3);
    check(null_search != NULL && strcmp(null_search, "[]") == 0, "NULL handle still returns []");
    astrum_memory_free_string(null_search);
    check(astrum_memory_node_count(NULL) == 0, "NULL handle counts zero");
    check(astrum_memory_enforce_capacity(NULL, 10) == 0, "NULL handle evicts nothing");
    check(astrum_memory_promote_confirmed(NULL, 1, 1) == 0, "NULL handle promotes nothing");
    check(astrum_memory_save(edge, "/nonexistent-dir/snap.json") != 0, "save to a bad path fails cleanly");
    check(astrum_memory_load("/nonexistent-dir/snap.json") == NULL, "load of a missing file returns NULL");
    astrum_memory_free_string(NULL);   /* documented no-op */
    astrum_memory_destroy(NULL);       /* documented no-op */
    astrum_memory_destroy(edge);

    if (failures == 0) {
        printf("ALL OK\n");
        return 0;
    }
    printf("%d CHECK(S) FAILED\n", failures);
    return 1;
}
