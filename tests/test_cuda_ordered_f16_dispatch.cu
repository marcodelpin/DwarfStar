/* Regression test for the ordered-F16 dispatch decision on Blackwell parts.
 *
 * WHY IT EXISTS: nothing under tests/ varied DS4_CUDA_FORCE_ORDERED_F16_MATMUL,
 * so the override precedence and the architecture bound were unguarded. Both are
 * easy to get wrong in a way that compiles, passes every other test, and silently
 * changes the default kernel on somebody's GPU.
 *
 * WHAT IT COVERS: cuda_arch_ordered_f16_measured() and cuda_skip_ordered_f16_matmul(),
 * the REAL functions -- this TU includes ds4_cuda.cu rather than restating the logic,
 * so a copy cannot drift away from the original.
 *
 * WHAT IT DOES NOT COVER: the init-time veto in ds4_gpu_init_multi() that computes
 * g_cuda_ordered_f16_skip across devices. Reaching it needs cudaGetDeviceProperties
 * and therefore real hardware, and a mixed-architecture rig to make it discriminating.
 * This test drives the flag directly instead; the veto wiring itself is unverified.
 *
 * Runs entirely on the host: no CUDA API is called on any path exercised here.
 */

#include "../ds4_cuda.cu"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;
static int g_checks = 0;

static void check(int got, int want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        printf("  FAIL %-58s got=%d want=%d\n", what, got, want);
    } else {
        printf("  ok   %-58s = %d\n", what, got);
    }
}

static void clear_env(void) {
    unsetenv("DS4_CUDA_FORCE_ORDERED_F16_MATMUL");
    unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
}

int main(void) {
    printf("== cuda_arch_ordered_f16_measured: only majors with a published measurement ==\n");
    /* sm_110 (Jetson AGX Thor) and sm_121 (GB10) are the two with data. */
    check(cuda_arch_ordered_f16_measured(11), 1, "major 11 (sm_110, Thor) is measured");
    check(cuda_arch_ordered_f16_measured(12), 1, "major 12 (sm_121, GB10) is measured");
    /* The bound is the point of the test: an unbounded `>= 11` would opt every
     * future major into a default nothing has measured on it. */
    check(cuda_arch_ordered_f16_measured(13), 0, "major 13 (future) is NOT assumed");
    check(cuda_arch_ordered_f16_measured(99), 0, "major 99 (far future) is NOT assumed");
    check(cuda_arch_ordered_f16_measured(10), 0, "major 10 (older) keeps the default");
    check(cuda_arch_ordered_f16_measured(8),  0, "major 8 (Ampere) keeps the default");
    check(cuda_arch_ordered_f16_measured(0),  0, "major 0 (unknown) keeps the default");

    printf("\n== cuda_skip_ordered_f16_matmul: the flag decides when no override is set ==\n");
    clear_env();
    g_cuda_ordered_f16_skip = 1;
    check(cuda_skip_ordered_f16_matmul(), 1, "flag=1, no env -> skip");
    g_cuda_ordered_f16_skip = 0;
    check(cuda_skip_ordered_f16_matmul(), 0, "flag=0, no env -> do not skip");

    printf("\n== overrides beat the flag, in both directions ==\n");
    clear_env();
    setenv("DS4_CUDA_FORCE_ORDERED_F16_MATMUL", "1", 1);
    g_cuda_ordered_f16_skip = 1;
    check(cuda_skip_ordered_f16_matmul(), 0, "FORCE set, flag=1 -> do not skip");
    g_cuda_ordered_f16_skip = 0;
    check(cuda_skip_ordered_f16_matmul(), 0, "FORCE set, flag=0 -> do not skip");

    clear_env();
    setenv("DS4_CUDA_NO_ORDERED_F16_MATMUL", "1", 1);
    g_cuda_ordered_f16_skip = 0;
    check(cuda_skip_ordered_f16_matmul(), 1, "NO set, flag=0 -> skip (upstream meaning kept)");
    g_cuda_ordered_f16_skip = 1;
    check(cuda_skip_ordered_f16_matmul(), 1, "NO set, flag=1 -> skip");

    printf("\n== FORCE wins over NO (it is tested first, and that ordering is load-bearing) ==\n");
    clear_env();
    setenv("DS4_CUDA_FORCE_ORDERED_F16_MATMUL", "1", 1);
    setenv("DS4_CUDA_NO_ORDERED_F16_MATMUL", "1", 1);
    g_cuda_ordered_f16_skip = 1;
    check(cuda_skip_ordered_f16_matmul(), 0, "FORCE+NO both set -> FORCE wins, do not skip");

    printf("\n== an empty value still counts as set (getenv != NULL, not a truthiness test) ==\n");
    clear_env();
    setenv("DS4_CUDA_FORCE_ORDERED_F16_MATMUL", "", 1);
    g_cuda_ordered_f16_skip = 1;
    check(cuda_skip_ordered_f16_matmul(), 0, "FORCE=\"\" -> still an override");
    clear_env();
    setenv("DS4_CUDA_NO_ORDERED_F16_MATMUL", "0", 1);
    g_cuda_ordered_f16_skip = 0;
    check(cuda_skip_ordered_f16_matmul(), 1, "NO=\"0\" -> still an override (matches upstream)");

    clear_env();
    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    if (g_checks < 16) {
        printf("SUITE DID NOT RUN ITS FULL BODY (expected at least 16 checks)\n");
        return 2;
    }
    printf(g_failures ? "FAILED\n" : "PASSED\n");
    return g_failures ? 1 : 0;
}
