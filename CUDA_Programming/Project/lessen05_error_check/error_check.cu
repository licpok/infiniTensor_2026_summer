#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
    return error;
}

int main() {

    // 1. 分配主机内存并初始化
    float *fpHost_A;
    fpHost_A = (float*)malloc(4);
    memset(fpHost_A, 0, 4);

    // 2. 分配设备内存并初始化
    float *fpDevice_A;
    cudaError_t error = ErrorCheck(cudaMalloc((void **)&fpDevice_A, 4), __FILE__, __LINE__);// 此处__FILE__和__LINE__ 均为预定义宏
    cudaMemset(fpDevice_A, 0, 4);

    // 3. 数据从主机复制到设备
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    // 此处传参错误，报错信息：CUDA Error: invalid argument in error_check.cu at line 25

    free(fpHost_A);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);

    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}