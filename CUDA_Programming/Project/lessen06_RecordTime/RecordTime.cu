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
    dim3 block(32);
    dim3 grid(iElemCount / 32); // 512 / 32

    // ==========================================
    // 1. 初始化事件句柄与时间累加器
    // ==========================================
    // 定义两个 CUDA 事件变量，分别作为计时的“起点”和“终点”标记
    cudaEvent_t start, stop;
    // 创建 start 事件，并在内部为该事件对象分配 GPU 底层资源（附带宏定义的错误检查）
    ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    // 创建 stop 事件，同样在内部为该事件对象分配 GPU 底层资源
    ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    // 初始化总时间累计变量，用于后续累加各次迭代的运行耗时（单位：毫秒）
    float t_sum = 0;
    // ==========================================
    // 2. 开始循环测试（迭代 11 次以获取稳定值）
    // ==========================================
    for (int repeat = 0; repeat <= 10; repeat++) {
        // 向默认 CUDA 流中异步插入一个“起始时间戳”标记。当 GPU 执行到此处时会记录当前时间
        ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
        // 异步启动 GPU 核函数。此时 CPU 仅负责将计算任务分发给 GPU 命令队列，随后立即向下执行
        addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
        // 向默认 CUDA 流中异步插入一个“结束时间戳”标记。它会在上述核函数完全执行完毕后被 GPU 记录
        ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
        // 【核心同步点】阻塞 CPU 线程，强制等待 GPU 把 stop 事件之前的所有任务（包括核函数）彻底执行完
        ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
        // 定义一个局部变量，用于接收当前单次循环的耗时结果
        float elapsed_time;
        // 计算 start 和 stop 两个标记之间的时间差，结果存入 elapsed_time，单位为毫秒（精度达微秒级）
        ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
        // 剔除第 0 次（第一次）的“冷启动”时间。只有当 repeat 大于 0 时，才将稳定期的耗时累加进总时间
        if (repeat > 0) t_sum += elapsed_time;
    }
    // ==========================================
    // 3. 善后处理与资源释放
    // ==========================================
    // 销毁 start 事件，释放该事件在 GPU 内部占用的全部硬件和驱动资源
    ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
    // 销毁 stop 事件，释放该事件在 GPU 内部占用的全部硬件和驱动资源
    ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    // 计算排除冷启动后的 10 次有效运行的平均时间（单位：毫秒）
    float avg_time = t_sum / 10.0f;
    // 输出平均时间
    printf("平均耗时 : %f\n", avg_time);

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
