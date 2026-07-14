#include <stdio.h>

/*
线程模型结构

两大核心：grid(网格)，block(线程块)

线程划分是逻辑结构，物理上是线性结构

每个线程在核函数中有唯一身份标识即索引(可依靠内建变量计算)
以默认维度(一维)为例：grid和block均为dim3变量，都具有x,y,z三个维度，默认形况下y,z均为1
(1) gridDim.x: grid_size 的值 (gridDim 为uint3变量，有x,y,z三个变量)
(2) blockDim.x: block_size 的值 (blockDim 为uint3变量，有x,y,z三个变量)
(3) 线程唯一标识 Idx = threadIdx.x + blockIdx.x * blockDim.x

网格大小限制：
gridDim.x最大值为2^31-1
gridDim.y和gridDim.z最大值为2^16-1

线程块大小限制：
blockDim.x和blockDim.y最大值为1024
blockDim.z最大值为64
注：线程块总的大小(blockDim.x * blockDim.y * blockDim.z)不能超过1024
*/

__global__ void say_hello() {
    const int bid = blockIdx.x;  // blockIdx
    const int tid = threadIdx.x; // threadIdx

    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hello CUDA from thread %d, block %d, id %d  \n", tid, bid, id);
}

int main() {

    // dim3构造函数
    dim3 grid_size(3, 2, 1);
    dim3 block_size(5, 3, 1);

    // <<<grid_size, block_size>>> grid_size表示一个grid含有多少block,block_size表示一个block有多少thread
    say_hello<<<2, 4>>>(); // 默认是一维结构，即dim3 gridDim(grid_size,1,1), dim3 blockDim(block_size,1,1)
    // 同步CPU与GPU
    cudaDeviceSynchronize();

    /*
    线程计算：

    一维：
    Id = blockIdx.x * blockDim.x + threadIdx.x

    二维：
    blockId = blockIdx.x + blockIdx.y * gridDim.x
    threadId = threadIdx.x + threadIdx.y * blockDim.x
    Id = blockId * (blockDim.x + blockDim.y) + threadId

    三维：
    blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y)
    threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y)
    Id = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId
    */

    return 0;
}