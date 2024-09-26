#include <immintrin.h>
const char *dgemm_desc = "blocked dgemm with avx.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)8)
#endif

void print_double_array(__m512d arr, char a)
{
    double *temp = (double *)malloc(8 * sizeof(double));
    _mm512_storeu_pd(temp, arr);
    printf("%c ", a);
    for (int i = 0; i < 8; i++)
    {
        printf("%f ", temp[i]);
    }
    printf("\n");
}

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    if (M != BLOCK_SIZE || N != BLOCK_SIZE || K != BLOCK_SIZE)
    {
        int i, j, k;
        for (i = 0; i < M; ++i)
        {
            for (j = 0; j < N; ++j)
            {
                double cij = C[j * lda + i];
                for (k = 0; k < K; ++k)
                {
                    cij += A[k * lda + i] * B[j * lda + k];
                }
                C[j * lda + i] = cij;
            }
        }
    }
    else
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            __m512d c0 = _mm512_loadu_pd(&C[i * lda]);
            // print_double_array(c0,'i');

            for (int j = 0; j < 8; j++)
            {
                __m512d a = _mm512_set1_pd(B[i * lda + j]);
                // __m512d a = _mm512_broadcast_sd(&B[i * lda + j]);
                // print_double_array(a,'a');

                // 使用 AVX-512 加载 512 位数据 (8 doubles)
                __m512d b = _mm512_loadu_pd(&A[j * lda]);
                // __m512d b = _mm512_loadu_pd(&A[j * lda]);
                // print_double_array(b,'b');

                // 执行加法和乘法的计算
                c0 = _mm512_fmadd_pd(a, b, c0);  // c0 = a * b + c0
                // c0 = _mm512_add_pd(c0, _mm512_mul_pd(a, b));
                // print_double_array(c0,'e');
            }

            _mm512_storeu_pd(&C[i * lda], c0);
            // print_double_array(c0,'t');
        }
    }
}

void do_block(const int lda,
              const double *A, const double *B, double *C,
              const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE > lda ? lda - i : BLOCK_SIZE);
    const int N = (j + BLOCK_SIZE > lda ? lda - j : BLOCK_SIZE);
    const int K = (k + BLOCK_SIZE > lda ? lda - k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k * lda, B + k + j * lda, C + i + j * lda);
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi)
    {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj)
        {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk)
            {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
