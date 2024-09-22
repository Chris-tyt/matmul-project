const char *dgemm_desc = "My prefetch.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k, f_num;
    for (i = 0; i < M; ++i)
    {
        // for(f_num = 0;f_num<M;++f_num)
        // __builtin_prefetch(&A[f_num*M+i], 0, 2);  // pre-fetch A[i][j+M]
        for (j = 0; j < M; ++j)
        {
            __builtin_prefetch(&C[j * M + i], 0, 1);
            double cij = C[j * M + i];
            for (k = 0; k < M; ++k)
            {
                __builtin_prefetch(&A[k * M + i], 0, 1);
                __builtin_prefetch(&B[j * M + k], 0, 1);
                cij += A[k * M + i] * B[j * M + k];
            }
            __builtin_prefetch(&C[j * M + i], 1, 1);
            C[j * M + i] = cij;
        }
    }
}
