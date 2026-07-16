#include <stdio.h>

/*
核函数(Kernel function)
注意事项：
1. 核函数只能访问GPU显存
2. 核函数不能使用变长参数
3. 核函数不能使用静态变量
4. 核函数不能使用函数指针
5. 核函数具有异步性
*/ 

// __global__ 为kernel函数即核函数标志，是GPU内置函数，只能访问GPU显存，不能访问CPU内存
__global__ void say_hello() {
    // kernel函数中只能使用C语言风格输出
    printf("hello CUDA!\n");
}

int main() {

    // 调用kernel函数，需在 <<<x1, x2>>> 中指定线程参数(见lessen02)，
    say_hello<<<1, 1>>>();

    // 用于实现CPU与GPU的同步
    cudaDeviceSynchronize();

    return 0;
}