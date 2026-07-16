#include <stdio.h>

/*
虚拟架构与真实架构

虚拟架构：将C/C++源码编译成PTX，引入虚拟架构层，与硬件无关
意义：解决GPU版本更迭过快，导致版本不兼容问题
注意：虚拟架构版本不能超过真实架构版本，否则GPU无法执行任务

真实架构：将PTX编译成适配GPU的机器码，并在GPU上执行

虚拟架构计算能力和真实架构计算能力均可在编译时显式指定，且不能单独指定真实架构计算能力
虚拟架构计算能力：-arch=compute_XY (X为主版本号，Y为次版本号)
真实架构计算能力：-code=sm_XY (X为主版本号，Y为次版本号) X不同不能兼容

编译时也可指定多GPU版本编译(Fatbinary)
语法：-gencode=arch=compute_XY,code=sm_XY
注意：过多指定计算能力会增加编译时间和可执行文件大小，最后一行必须有打包了的低版本PTX
*/

__global__ void say_hello() {
    const int bid = blockIdx.x;  // blockIdx
    const int tid = threadIdx.x; // threadIdx

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hello CUDA from thread %d, block %d, id %d  \n", tid, bid, id);
}

int main() {

    // nvcc 可自动区分C/C++编译器编译的host代码，和nvcc编译的device代码
    printf("hello CUDA from CPU\n"); // CPU 执行
    say_hello<<<2, 4>>>(); // GPU 执行

    cudaDeviceSynchronize();

    /*
    本机默认虚拟架构计算能力为sm_52，最高可支持sm_87
    默认情况(可执行文件compute_xx) 选择适配本机的虚拟架构计算能力和真实架构计算能力
    指定虚拟架构(compute_xx30)计算能力较低时，编译会警告虚拟架构即将启用，但能正常运行
    指定虚拟架构计算能力较高时，编译正常通过，执行时只有CPU执行，GPU不执行
    当真实架构计算能力小于虚拟架构计算能力时，无法通过编译

    THINKING:
    compute_mult和compute_mult5的差异！！！(提示：即时编译JIT)
    ANSWER:
    即时编译指的是在运行可执行文件时，从保留的PTX代码临时编译出cubin文件
    指定方式：-gencode=arch=compute_XY,code=compute_XY 此处均为虚拟架构计算能力且必须一致
    */

    return 0;
}