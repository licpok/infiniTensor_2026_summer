#include <stdio.h>
#include "../tools/common.cuh"

void InitialData(float* addr, int elemCount){
    for (int i = 0; i < elemCount; i++) {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

__device__ float add(float A, float B) {
    return A + B;
}

__global__ void addFromGPU(float* A, float* B, float* C, const int N) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = bid * blockDim.x + tid;
    if(id < N) C[id] = add(A[id], B[id]); // if 语句非常重要
}

int main() {

    // 1. 设置GPU设备
    setGPU();

    // 2. 分配主机内存和设备内存，并初始化
    int iElemCount = 512;
    size_t stBytesCount = iElemCount * sizeof(float);
    // (1) 分配主机内存
    float *fpHost_A = (float *)malloc(stBytesCount);
    float *fpHost_B = (float *)malloc(stBytesCount);
    float *fpHost_C = (float *)malloc(stBytesCount);
    if(fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL) {
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_A, 0, stBytesCount);
    }
    else {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }
    // (2) 分配设备内存
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    cudaMalloc((float **)&fpDevice_A, stBytesCount);
    cudaMalloc((float **)&fpDevice_B, stBytesCount);
    cudaMalloc((float **)&fpDevice_C, stBytesCount);
    if(fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL) {
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_A, 0, stBytesCount);
        cudaMemset(fpDevice_A, 0, stBytesCount);
    }
    else {
        printf("Fail to allocate device memory!\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 3. 初始化主机中的数据
    srand(5201314);
    InitialData(fpHost_A, iElemCount);
    InitialData(fpHost_B, iElemCount);

    // 4. 数据从主机拷贝到设备
    cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);
    
    // 5. 调用核函数在设备中计算
    dim3 block(2048); // 2048个线程，超出最大线程数1024
    dim3 grid(iElemCount / 32); // 512 / 32
    addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
    ErrorCheck(cudaGetLastError(), __FILE__, __LINE__); // 检查核函数调用是否成功，获取同步前最后一个错误
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // 6. 将计算得到的数据从设备传给主机
    cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);

    // 7. 释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    cudaFree(fpDevice_A);
    cudaFree(fpDevice_B);
    cudaFree(fpDevice_C);

    cudaDeviceReset();

    return 0;
}
