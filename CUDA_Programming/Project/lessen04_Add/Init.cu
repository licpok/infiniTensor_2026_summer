#include <stdio.h>

/*
GPU加法带你理解GPU编程全流程
1. 设置GPU设备
2. 分配主机和设备内存
3. 初始化主机中的数据
4. 数据从主机复制到设备
5. 调用核函数在设备中进行计算
6. 将计算得到的数据从设备传到主机
7. 释放主机和设备内存
*/

int main() {

    // GPU设置
    // 获取GPU设备数量,返回cudaSuccess表示成功
    // 函数原型：__host____device__ cudaError_t cudaGetDevice(int* Count);
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
    if(error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found!\n");
    }
    else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    // 设置GPU执行时使用的设备
    // 函数原型：__host__ cudeError_t cudaSetDevice(int device);
    int iDev = 0;
    error = cudaSetDevice(iDev); 
    if(error != cudaSuccess) {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else {
        printf("set GPU 0 for computing.\n");
    }

    // 内存分配
    // 主机分配内存
    // 函数原型：extern void* malloc(unsigned int num_bytes);
    float *fpHost_A;
    fpHost_A = (float *)malloc(10);
    // 设备分配内存
    // 函数原型：__host____device__ cudaError_t cudaMalloc(void** devPtr, size_t size);
    float *fpDevice_A;
    cudaMalloc((float **)&fpDevice_A, 10);

    // 数据拷贝
    // 主机数据拷贝
    // 函数原型：void* memcpy(void* dest, const void* src, size_t n);
    // void* 表任意类型指针， dest 为目标地址， src 为源地址， n表拷贝字节数
    int *a = (int *)malloc(10);
    int *b = (int *)malloc(10);
    memcpy(a, b, 10);
    // 设备数据拷贝
    // 函数原型：__host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
    // 最后一位参数kind表示四种拷贝方向，cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice

    // 内存初始化
    // 主机内存初始化
    // 函数原型：void* memset(void* str, int c, size_t n);
    memset(fpHost_A, 0, 10);
    // 设备内存初始化
    // 函数原型：__host__ cudaError_t cudaMemset(void* devPtr, int value, size_t count);
    cudaMemset(fpDevice_A, 0, 10);

    // 内存释放
    // 主机内存释放
    free(fpHost_A);
    // 设备内存释放
    // 函数原型：__host____device__ cudaError_t cudaFree(void* devPtr);
    cudaFree(fpDevice_A);

    return 0;
}