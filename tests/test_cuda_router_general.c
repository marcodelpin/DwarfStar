/* Regression test for router selection at shapes other than 256 experts
 * (upstream issue #427).
 *
 * WHY IT EXISTS: ds4_gpu_router_select_tensor and ds4_gpu_router_select_batch_tensor
 * returned 0 unless the model had exactly 256 experts, 6 used and a weight
 * scale of 1.5.  On CUDA that is not a slow path but no path: the caller in
 * ds4.c turns the 0 into ok=false for the whole decode graph, so a 384-expert
 * model could not run on CUDA at all.  Nothing under tests/ ever called these
 * entry points with another shape, so the limit was invisible.
 *
 * WHAT IT COVERS: both entry points, through the public ABI, at the shape they
 * always served (256/6/1.5, still handled by the untouched fixed-size kernels)
 * AND at shapes that now reach router_select_general_kernel.
 *
 * WHY THE 256 CASES ARE NOT DECORATION: they validate the host reference
 * against kernels that are known good and are not touched by this change.  A
 * reference that agrees there has earned the right to judge the general path;
 * one that only ever ran against the new kernel would be a restatement of it.
 * The reference itself is written as "sort by (score desc, index asc), take
 * the first k" -- an independent formulation of the same contract, not a copy
 * of the kernel's insertion loop.
 *
 * Needs a GPU: this exercises real kernels.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The device is built with --use_fast_math, so expf/log1pf/sqrtf there are the
 * lower-accuracy intrinsics and cannot be expected to agree with host libm to
 * the last ulp.  A few 1e-4 is still four orders of magnitude tighter than any
 * wrong-algorithm difference would be.  Selection is compared EXACTLY: the
 * ordering is a discrete decision, and the patterns below keep the top-k well
 * separated except where they tie exactly, which both sides resolve by index. */
#define ROUTER_TOL 2.0e-4f

static int g_failures = 0;
static int g_checks = 0;

/* One model map for the whole run, allocated once and never freed, with a
 * distinct 256 KiB slot per case.  cuda_model_range_ptr() caches device
 * mappings keyed on the host base pointer and may cudaHostRegister the pages,
 * so a per-case malloc/free could hand a later case a stale mapping of a
 * recycled address, and freeing a registered page is undefined behaviour.
 * Slots are far larger than any page size on any host we build for, so no two
 * cases can share a registered page either. */
#define TEST_MODEL_SLOT (256u * 1024u)
#define TEST_MODEL_SLOTS 32u
static unsigned char *g_model_base = NULL;
static uint32_t g_slot = 0;

static void check(int got, int want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        printf("  FAIL %-62s got=%d want=%d\n", what, got, want);
    } else {
        printf("  ok   %-62s = %d\n", what, got);
    }
}

/* Byte-for-byte what softplus_dev does on the device. */
static float softplus_ref(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

typedef struct { float score; uint32_t idx; } ranked;

static int ranked_cmp(const void *a, const void *b) {
    const ranked *x = (const ranked *)a;
    const ranked *y = (const ranked *)b;
    if (x->score > y->score) return -1;
    if (x->score < y->score) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

/* Host reference for one token. hash_row != NULL selects hash mode. */
static void reference_router(const float *logits,
                             const float *bias,
                             const int32_t *hash_row,
                             uint32_t n_expert,
                             uint32_t n_expert_used,
                             float expert_weight_scale,
                             float *out_probs,
                             int32_t *out_sel,
                             float *out_w) {
    ranked *r = (ranked *)malloc((size_t)n_expert * sizeof(ranked));
    if (!r) return;
    for (uint32_t i = 0; i < n_expert; i++) {
        out_probs[i] = sqrtf(softplus_ref(logits[i]));
        r[i].score = out_probs[i] + (bias ? bias[i] : 0.0f);
        r[i].idx = i;
    }
    if (hash_row) {
        for (uint32_t j = 0; j < n_expert_used; j++) out_sel[j] = hash_row[j];
    } else {
        qsort(r, n_expert, sizeof(ranked), ranked_cmp);
        for (uint32_t j = 0; j < n_expert_used; j++) out_sel[j] = (int32_t)r[j].idx;
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        const int32_t e = out_sel[j];
        const float v = (e >= 0 && (uint32_t)e < n_expert) ? out_probs[e] : 0.0f;
        out_w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < n_expert_used; j++) out_w[j] = out_w[j] / sum * expert_weight_scale;
    free(r);
}

/* pattern 0: well separated. pattern 1: many exact ties, so the index
 * tiebreak is exercised rather than assumed. */
static float logit_pattern(uint32_t e, uint32_t t, uint32_t pattern) {
    if (pattern == 1u) return ((e + t) % 5u == 0u) ? 3.0f : -1.0f;
    const uint32_t h = ((e + 1u) * 2654435761u) ^ ((t + 7u) * 40503u);
    return (float)((int32_t)(h % 4001u) - 2000) / 256.0f;
}

/* One shape, one mode, through whichever entry point n_tokens selects. */
static void run_case(const char *name,
                     uint32_t n_expert,
                     uint32_t n_expert_used,
                     float expert_weight_scale,
                     uint32_t n_tokens,
                     int has_bias,
                     int hash_mode,
                     uint32_t pattern) {
    const uint32_t hash_rows = 8;
    const uint64_t logit_count = (uint64_t)n_tokens * n_expert;
    const uint64_t sel_count = (uint64_t)n_tokens * n_expert_used;
    /* Within this case's slot: bias at 0, hash table half a slot later. */
    const uint64_t hash_offset = TEST_MODEL_SLOT / 2u;
    const uint64_t model_bytes = TEST_MODEL_SLOT;

    unsigned char *model = g_model_base + (uint64_t)g_slot * TEST_MODEL_SLOT;
    g_slot++;
    float *logits_host = (float *)malloc((size_t)logit_count * sizeof(float));
    float *probs_host = (float *)malloc((size_t)logit_count * sizeof(float));
    float *ref_probs = (float *)malloc((size_t)n_expert * sizeof(float));
    int32_t *sel_host = (int32_t *)malloc((size_t)sel_count * sizeof(int32_t));
    float *w_host = (float *)malloc((size_t)sel_count * sizeof(float));
    int32_t *ref_sel = (int32_t *)malloc((size_t)n_expert_used * sizeof(int32_t));
    float *ref_w = (float *)malloc((size_t)n_expert_used * sizeof(float));
    int32_t *tok_host = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    if (g_slot > TEST_MODEL_SLOTS || !logits_host || !probs_host || !ref_probs || !sel_host ||
        !w_host || !ref_sel || !ref_w || !tok_host) {
        printf("  FAIL %-62s host allocation\n", name);
        g_checks++; g_failures++;
        goto done;
    }

    memset(model, 0, (size_t)model_bytes);
    float *bias = (float *)model;
    for (uint32_t i = 0; i < n_expert; i++) bias[i] = ((i % 37u) == 3u) ? 4.0f : -0.05f * (float)(i % 11u);
    int32_t *hash = (int32_t *)(model + hash_offset);
    for (uint32_t r = 0; r < hash_rows; r++) {
        for (uint32_t j = 0; j < n_expert_used; j++) {
            hash[r * n_expert_used + j] = (int32_t)((r * 13u + j * 29u) % n_expert);
        }
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        tok_host[t] = (int32_t)(t % hash_rows);
        for (uint32_t e = 0; e < n_expert; e++) {
            logits_host[(uint64_t)t * n_expert + e] = logit_pattern(e, t, pattern);
        }
    }

    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logit_count * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(logit_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(sel_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sel_count * sizeof(float));
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc((uint64_t)n_tokens * sizeof(int32_t));
    int accepted = -1;
    if (logits && probs && selected && weights && tokens &&
        ds4_gpu_tensor_write(logits, 0, logits_host, logit_count * sizeof(float)) &&
        ds4_gpu_tensor_write(tokens, 0, tok_host, (uint64_t)n_tokens * sizeof(int32_t))) {
        if (n_tokens == 1u) {
            accepted = ds4_gpu_router_select_tensor(
                           selected, weights, probs, model, model_bytes,
                           0, hash_offset, hash_rows, (uint32_t)tok_host[0],
                           n_expert, n_expert_used, expert_weight_scale, 1, 0,
                           has_bias ? true : false, hash_mode ? true : false,
                           logits) ? 1 : 0;
        } else {
            accepted = ds4_gpu_router_select_batch_tensor(
                           selected, weights, probs, model, model_bytes,
                           0, hash_offset, hash_rows, 1, 0,
                           has_bias ? true : false, hash_mode ? true : false,
                           logits, tokens,
                           n_expert, n_expert_used, expert_weight_scale, n_tokens) ? 1 : 0;
        }
        if (accepted) (void)ds4_gpu_synchronize();
    }
    check(accepted, 1, name);

    if (accepted == 1 &&
        ds4_gpu_tensor_read(probs, 0, probs_host, logit_count * sizeof(float)) &&
        ds4_gpu_tensor_read(selected, 0, sel_host, sel_count * sizeof(int32_t)) &&
        ds4_gpu_tensor_read(weights, 0, w_host, sel_count * sizeof(float))) {
        uint64_t sel_bad = 0, w_bad = 0, p_bad = 0;
        for (uint32_t t = 0; t < n_tokens; t++) {
            reference_router(logits_host + (uint64_t)t * n_expert,
                             (has_bias && !hash_mode) ? bias : NULL,
                             hash_mode ? hash + (uint32_t)tok_host[t] * n_expert_used : NULL,
                             n_expert, n_expert_used, expert_weight_scale,
                             ref_probs, ref_sel, ref_w);
            for (uint32_t e = 0; e < n_expert; e++) {
                const float got = probs_host[(uint64_t)t * n_expert + e];
                if (fabsf(got - ref_probs[e]) > ROUTER_TOL * (1.0f + fabsf(ref_probs[e]))) p_bad++;
            }
            for (uint32_t j = 0; j < n_expert_used; j++) {
                if (sel_host[(uint64_t)t * n_expert_used + j] != ref_sel[j]) sel_bad++;
                const float got = w_host[(uint64_t)t * n_expert_used + j];
                if (fabsf(got - ref_w[j]) > ROUTER_TOL * (1.0f + fabsf(ref_w[j]))) w_bad++;
            }
        }
        char what[192];
        snprintf(what, sizeof(what), "%s: probs match", name);
        check(p_bad == 0 ? 1 : 0, 1, what);
        snprintf(what, sizeof(what), "%s: selection matches", name);
        check(sel_bad == 0 ? 1 : 0, 1, what);
        snprintf(what, sizeof(what), "%s: weights match", name);
        check(w_bad == 0 ? 1 : 0, 1, what);
    } else if (accepted == 1) {
        char what[192];
        snprintf(what, sizeof(what), "%s: readback", name);
        check(0, 1, what);
    }

    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(logits);
done:
    free(tok_host); free(ref_w); free(ref_sel); free(w_host);
    free(sel_host); free(ref_probs); free(probs_host); free(logits_host);
}

/* Shapes that must still be refused. Each one is a real out-of-bounds if it
 * were let through, so "declined" is the safety property, not a limitation. */
static void run_refusal(const char *name,
                        uint32_t n_expert,
                        uint32_t n_expert_used,
                        uint64_t probs_bytes_override) {
    const uint32_t alloc_expert = n_expert ? n_expert : 1u;
    const uint32_t alloc_used = n_expert_used ? n_expert_used : 1u;
    float sinks_model[8] = {0};
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc((uint64_t)alloc_expert * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(probs_bytes_override ? probs_bytes_override
                                                                     : (uint64_t)alloc_expert * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)alloc_used * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)alloc_used * sizeof(float));
    int accepted = -1;
    if (logits && probs && selected && weights) {
        accepted = ds4_gpu_router_select_tensor(
                       selected, weights, probs, sinks_model, sizeof(sinks_model),
                       0, 0, 1, 0, n_expert, n_expert_used, 1.5f, 1, 0,
                       false, false, logits) ? 1 : 0;
    }
    check(accepted, 0, name);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(probs);
    ds4_gpu_tensor_free(logits);
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed -- this test needs a CUDA device\n");
        return 2;
    }
    if (posix_memalign((void **)&g_model_base, 65536,
                       (size_t)TEST_MODEL_SLOT * TEST_MODEL_SLOTS) != 0 || !g_model_base) {
        fprintf(stderr, "model map allocation failed\n");
        return 2;
    }
    memset(g_model_base, 0, (size_t)TEST_MODEL_SLOT * TEST_MODEL_SLOTS);

    printf("== control: the shape the fixed-size kernels serve (untouched by this change) ==\n");
    run_case("256/6/1.5 single, no bias",      256, 6, 1.5f, 1, 0, 0, 0);
    run_case("256/6/1.5 single, bias",         256, 6, 1.5f, 1, 1, 0, 0);
    run_case("256/6/1.5 single, ties",         256, 6, 1.5f, 1, 0, 0, 1);
    run_case("256/6/1.5 batch x5, bias",       256, 6, 1.5f, 5, 1, 0, 0);
    run_case("256/6/1.5 batch x5, hash",       256, 6, 1.5f, 5, 0, 1, 0);

    printf("\n== general shapes: refused outright before this change ==\n");
    run_case("384/8/2.5 single, no bias",      384, 8, 2.5f, 1, 0, 0, 0);
    run_case("384/8/2.5 single, bias",         384, 8, 2.5f, 1, 1, 0, 0);
    run_case("384/8/2.5 single, ties",         384, 8, 2.5f, 1, 1, 0, 1);
    run_case("384/8/2.5 batch x5, bias",       384, 8, 2.5f, 5, 1, 0, 0);
    run_case("384/8/2.5 batch x5, hash",       384, 8, 2.5f, 5, 0, 1, 0);
    run_case("128/4/1.0 single, bias",         128, 4, 1.0f, 1, 1, 0, 0);
    run_case("512/1/1.5 single, no bias",      512, 1, 1.5f, 1, 0, 0, 0);
    run_case("256/6/2.0 single, bias (scale only)", 256, 6, 2.0f, 1, 1, 0, 0);

    printf("\n== shapes that must still be refused ==\n");
    run_refusal("n_expert=0 refused",                  0, 6, 0);
    run_refusal("n_expert_used=0 refused",           256, 0, 0);
    run_refusal("n_expert_used>n_expert refused",      8, 9, 0);
    run_refusal("n_expert>4096 refused",            8192, 6, 0);
    run_refusal("undersized probs buffer refused",   384, 8, 64);

    ds4_gpu_cleanup();
    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_checks < 30) {
        printf("SUITE DID NOT RUN ITS FULL BODY (expected at least 30 checks)\n");
        return 2;
    }
    printf(g_failures ? "FAILED\n" : "PASSED\n");
    return g_failures ? 1 : 0;
}
