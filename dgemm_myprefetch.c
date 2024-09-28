#include <immintrin.h>
const char *dgemm_desc = "My prefetch.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k, f_num;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < M; ++j)
        {
            double cij = C[j * M + i];
            
            _mm_prefetch(&B[j*M],_MM_HINT_T0);
            for (k = 0; k < M; ++k)
            {
                cij += A[k * M + i] * B[j * M + k];
            }
            C[j * M + i] = cij;
        }
    }
}
