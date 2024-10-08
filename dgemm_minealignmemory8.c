#include <xmmintrin.h>

#define ALIAN_MEMORY ;
const char *dgemm_desc = "Align 8 bytes.";
const int alain_bits = 8;

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < M; ++j)
        {
            double cij = C[j * M + i];
            for (k = 0; k < M; ++k)
                cij += A[k * M + i] * B[j * M + k];
            C[j * M + i] = cij;
        }
    }
}
