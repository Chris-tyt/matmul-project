#include <immintrin.h>
const char *dgemm_desc = "Simple blocked dgemm.";

#define ALIAN_MEMORY ;
const int alain_bits = 64;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int)4)
#endif

void print_double_array(__m256d arr, char a)
{
    double *temp = (double *)malloc(4 * sizeof(double));
    _mm256_storeu_pd(temp, arr);
    printf("%c ", a);
    for (int i = 0; i < 4; i++)
    {
        printf("%f ", temp[i]); // 使用 %f 格式化符号来打印 double 类型
    }
    printf("\n"); // 换行
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
            // 初始化寄存器 C[i][*] 为 0
            // __m256d c0 = _mm256_setzero_pd();
            __m256d c0 = _mm256_loadu_pd(&C[i * lda]);

            // print_double_array(c0, 'i');
            for (int j = 0; j < BLOCK_SIZE; j++)
            {
                // 将 A[i][j] 的值广播到 256 位寄存器中 (每个寄存器能装载 4 个 double)
                __m256d a = _mm256_broadcast_sd(&B[i * lda + j]);
                // print_double_array(a, 'a');
                // 加载 B[j][*] 的 4 个 double 浮点数
                __m256d b = _mm256_loadu_pd(&A[j * lda]);
                // print_double_array(b, 'b');
                // 将 A[i][j] * B[j][*] 并行计算并累加到结果中
                c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));
                // print_double_array(c0, 'e');

                
            }


            // __m256d a = _mm256_broadcast_sd(&B[i * lda + 0]);
            // __m256d b = _mm256_loadu_pd(&A[0 * lda]);
            // c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));

            // a = _mm256_broadcast_sd(&B[i * lda + 1]);
            // b = _mm256_loadu_pd(&A[1 * lda]);
            // c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));

            // a = _mm256_broadcast_sd(&B[i * lda + 2]);
            // b = _mm256_loadu_pd(&A[2 * lda]);
            // c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));

            // a = _mm256_broadcast_sd(&B[i * lda + 3]);
            // b = _mm256_loadu_pd(&A[3 * lda]);
            // c0 = _mm256_add_pd(c0, _mm256_mul_pd(a, b));

            // 将结果存回 C[i][*]
            _mm256_storeu_pd(&C[i * lda], c0);

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
                // printf("----------------------\n");
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
