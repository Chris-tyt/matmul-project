const char* dgemm_desc = "unroll 4.";

void square_dgemm(const int M, 
                  const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M-4; k+=4){
                cij += A[k*M+i] * B[j*M+k];
                cij += A[(k+1)*M+i] * B[j*M+k+1];
                cij += A[(k+2)*M+i] * B[j*M+k+2];
                cij += A[(k+3)*M+i] * B[j*M+k+3];
            }
            for(;k<M;k++){
                cij += A[k*M+i] * B[j*M+k];
            }
            C[j*M+i] = cij;
        }
    }
}
