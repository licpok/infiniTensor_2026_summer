#pragma once
#include <stdlib.h>
#include <stdio.h>

void setGPU() {
    // 检测GPU数量
    int iDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
    if(error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if(error != cudaSuccess) {
        printf("fail to set GPU 0 for computing.\n");
        exit(-1);
    }
    else {
        printf("set GPU 0 for computing.\n");
    }
}

cudaError_t ErrorCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
    return error;
}