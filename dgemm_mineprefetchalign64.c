#include <xmmintrin.h>

#define ALIAN_MEMORY ;
const char *dgemm_desc = "prefetch and align 64 bytes.";
const int align_bytes = 64;

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k, f_num;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < M; ++j)
        {
            __builtin_prefetch(&B[j * M + k], 0, 1);
            double cij = C[j * M + i];
            for (k = 0; k < M; ++k)
            {
                cij += A[k * M + i] * B[j * M + k];
            }
            C[j * M + i] = cij;
        }
    }
}
