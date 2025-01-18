1, CPU里的scheduling是分时共享，GPU里的scheduling调度的最小单位是什么？是否也是分时共享
2，为什么又Grid的概念？它的作用是什么？Grid内的Block之间是怎么交互的？
3，计算thread Idx:
    i = blockIdx.x * blockDim.x + threadIdx.x
4,Warp, GPU是SIMT架构，同一个Warp内的thread并行执行同一条指令，with不同的数据
5，为什么block内的thread数量要求要一致给hi？不能够按照任务资源需要来分配？
6，Block是否可以跨SM？
7，如何决定block在哪个SM上执行？block内的thread是如何分配给warp的？以及这些调度是硬件层面做的，还是软件实现？
8，PTX指的是什么？