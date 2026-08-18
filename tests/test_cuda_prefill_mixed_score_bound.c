/* Regression test for the score-buffer bound of the CUDA mixed-prefill scalar
 * fallback (upstream issue #803).
 *
 * WHY IT EXISTS: attention_prefill_mixed_kernel keeps raw-window plus visible
 * compressed scores in a fixed shared array, and attention_prefill_mixed_launch
 * used to launch it for any shape at all.  The out-of-bounds write is silent:
 * it corrupts the neighbouring shared variables of the same block, so the run
 * completes and returns plausible numbers.  Nothing under tests/ noticed.
 *
 * WHAT IT COVERS: the launcher's decision, through the public entry point
 * ds4_gpu_attention_prefill_masked_mixed_heads_tensor -- not a restatement of
 * the bound.  Every case is a PAIR either side of the boundary, so a guard that
 * declined everything would fail just as loudly as one that declined nothing.
 * The accepted cases are also checked numerically against a host reference, so
 * "accepted" means "and it computed the right answer".
 *
 * WHY THE MASKED ENTRY AND head_dim != 512: the launcher tries a token-tile
 * path, a windowed kernel and a cuBLAS path before the scalar kernel.  The
 * first two require an unmasked call, all three require head_dim == 512.  A
 * masked call with head_dim 64 therefore reaches the scalar fallback with no
 * environment variable to set, which is one less thing that can silently stop
 * being true.
 *
 * Needs a GPU: this exercises real kernels, not a host-side predicate.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

/* The attention sinks double as the model map. It is allocated ONCE and never
 * freed on purpose: cuda_model_range_ptr() caches device mappings keyed on the
 * host pointer and may cudaHostRegister the page, so a per-case malloc/free
 * could hand a later case a stale mapping of a recycled address. */
#define TEST_N_HEAD 4u
static float g_sinks[TEST_N_HEAD];

static void check(int got, int want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        printf("  FAIL %-56s got=%d want=%d\n", what, got, want);
    } else {
        printf("  ok   %-56s = %d\n", what, got);
    }
}

/* Deterministic, spread over several orders of magnitude so a wrong softmax
 * maximum shows up rather than cancelling. */
static float pattern(uint32_t a, uint32_t b) {
    const uint32_t h = (a * 2654435761u) ^ (b * 40503u);
    return (float)((int32_t)(h % 2001u) - 1000) / 512.0f;
}

/* Host reference for attention_prefill_mixed_kernel. */
static void reference_mixed(float *out,
                            const float *sinks,
                            const float *q,
                            const float *raw_kv,
                            const float *comp_kv,
                            const float *comp_mask,
                            uint32_t n_tokens,
                            uint32_t n_comp,
                            uint32_t window,
                            uint32_t ratio,
                            uint32_t n_head,
                            uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc(((size_t)n_tokens + n_comp) * sizeof(float));
    if (!scores) return;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_start = (window != 0u && t + 1u > window) ? t + 1u - window : 0u;
        const uint32_t raw_count = t + 1u - raw_start;
        uint32_t visible_comp = (t + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float max_s = sinks[h];
            for (uint32_t r = 0; r < raw_count; r++) {
                const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++) dot += (double)qh[d] * kv[d];
                scores[r] = (float)dot * scale;
                if (scores[r] > max_s) max_s = scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float add = comp_mask[(uint64_t)t * n_comp + c];
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = comp_kv + (uint64_t)c * head_dim;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; d++) dot += (double)qh[d] * kv[d];
                    s = (float)dot * scale + add;
                }
                scores[raw_count + c] = s;
                if (s > max_s) max_s = s;
            }
            double denom = exp((double)sinks[h] - max_s);
            for (uint32_t i = 0; i < raw_count + visible_comp; i++) {
                scores[i] = (float)exp((double)scores[i] - max_s);
                denom += scores[i];
            }
            float *oh = out + ((uint64_t)t * n_head + h) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (uint32_t r = 0; r < raw_count; r++) {
                    acc += (double)raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
                }
                for (uint32_t c = 0; c < visible_comp; c++) {
                    acc += (double)comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
                }
                oh[d] = (float)(acc / denom);
            }
        }
    }
    free(scores);
}

/* Runs one shape through the masked mixed entry point.
 *
 * want_accept: 1 = the launcher must take it (and be numerically right),
 *              0 = it must decline, leaving the call to the CPU path.
 */
static void run_case(const char *name,
                     uint32_t n_tokens,
                     uint32_t n_comp,
                     uint32_t window,
                     uint32_t ratio,
                     int want_accept) {
    const uint32_t n_head = TEST_N_HEAD;
    const uint32_t head_dim = 64;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)(n_comp ? n_comp : 1u) * head_dim;
    const uint64_t mask_count = (uint64_t)n_tokens * (n_comp ? n_comp : 1u);

    float *sinks = g_sinks;
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    float *comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    float *mask_host = (float *)malloc((size_t)mask_count * sizeof(float));
    float *out_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *ref_host = (float *)malloc((size_t)q_count * sizeof(float));
    if (!q_host || !raw_host || !comp_host || !mask_host || !out_host || !ref_host) {
        printf("  FAIL %-56s host allocation\n", name);
        g_checks++;
        g_failures++;
        goto done;
    }
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = pattern((uint32_t)i, 1u);
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = pattern((uint32_t)i, 2u);
    for (uint64_t i = 0; i < comp_count; i++) comp_host[i] = pattern((uint32_t)i, 3u);
    /* Every fourth compressed column is masked out, so the -INFINITY branch of
     * the kernel is exercised and not just the plain one. */
    for (uint64_t i = 0; i < mask_count; i++) mask_host[i] = (i % 4u == 3u) ? -1.0e30f : 0.0f;
    memset(out_host, 0, (size_t)q_count * sizeof(float));

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *mask = ds4_gpu_tensor_alloc(mask_count * sizeof(float));
    int accepted = -1;
    if (heads && q && raw && comp && mask &&
        ds4_gpu_tensor_write(heads, 0, out_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(mask, 0, mask_host, mask_count * sizeof(float))) {
        accepted = ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
                       heads, sinks, (uint64_t)n_head * sizeof(float), 0,
                       q, raw, comp, 0, mask,
                       n_tokens, n_comp, window, ratio, n_head, head_dim) ? 1 : 0;
        if (accepted) (void)ds4_gpu_synchronize();
    }
    check(accepted, want_accept, name);

    if (accepted == 1 && want_accept == 1) {
        char what[160];
        if (!ds4_gpu_tensor_read(heads, 0, out_host, q_count * sizeof(float))) {
            snprintf(what, sizeof(what), "%s: readback", name);
            check(0, 1, what);
        } else {
            reference_mixed(ref_host, sinks, q_host, raw_host, comp_host, mask_host,
                            n_tokens, n_comp, window, ratio, n_head, head_dim);
            uint64_t bad = 0;
            float worst = 0.0f;
            for (uint64_t i = 0; i < q_count; i++) {
                const float d = fabsf(out_host[i] - ref_host[i]);
                const float tol = 2.0e-3f * (1.0f + fabsf(ref_host[i]));
                if (d > worst) worst = d;
                if (!(d <= tol)) bad++;
            }
            snprintf(what, sizeof(what), "%s: matches host reference (worst %.2e)",
                     name, (double)worst);
            check(bad == 0 ? 1 : 0, 1, what);
        }
    }

    ds4_gpu_tensor_free(mask);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
done:
    free(ref_host);
    free(out_host);
    free(mask_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed -- this test needs a CUDA device\n");
        return 2;
    }
    for (uint32_t h = 0; h < TEST_N_HEAD; h++) g_sinks[h] = 0.25f * (float)h - 0.5f;

    /* The cap is 512 scores.  Each pair below differs in exactly one input and
     * straddles it, so neither "decline everything" nor "decline nothing"
     * passes.  n_comp = 0 isolates the raw-window term. */
    printf("== raw-window term, boundary at 512 ==\n");
    run_case("n_tokens=512, window=0            -> accepted", 512, 0, 0, 4, 1);
    run_case("n_tokens=513, window=0            -> declined", 513, 0, 0, 4, 0);

    /* window caps raw_count, so a long prefill through a short window still
     * fits.  A guard that looked at n_tokens alone would wrongly decline this
     * and cost the CUDA path on every real windowed prefill. */
    printf("\n== window caps the raw term ==\n");
    run_case("n_tokens=2000, window=64          -> accepted", 2000, 0, 64, 4, 1);

    /* The compressed term counts too: same n_tokens, only ratio differs, and
     * a guard that ignored the compressed rows would accept both. */
    printf("\n== compressed term counts, ratio decides ==\n");
    run_case("n_tokens=500, window=100, ratio=2 -> accepted", 500, 1000, 100, 2, 1);
    run_case("n_tokens=500, window=100, ratio=1 -> declined", 500, 1000, 100, 1, 0);

    /* n_comp clamps the compressed term independently of ratio. */
    printf("\n== n_comp clamps the compressed term ==\n");
    run_case("n_tokens=500, window=100, n_comp=300 -> accepted", 500, 300, 100, 1, 1);

    ds4_gpu_cleanup();
    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_checks < 10) {
        printf("SUITE DID NOT RUN ITS FULL BODY (expected at least 10 checks)\n");
        return 2;
    }
    printf(g_failures ? "FAILED\n" : "PASSED\n");
    return g_failures ? 1 : 0;
}
