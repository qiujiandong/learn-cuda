# Coalesce

对比采用shared memory加速的结果和cublas实现的结果，实际上还是有很多优化空间。

有个新的概念是coalesced memory access，是为了最大化数据传输的带宽，尽可能使数据“联合”访问。

一个warp中的线程如果是访问global memory中的一块连续地址，那就是可以联合访问的。block的维度有一维，二维，三维，这是为了更好地与具体应用进行映射，而block的性能，只和block的size有关。也就是说8x2的block和4x4的block本质上是一样的，都是16个线程。每个线程有自己的ID，类似于二维数组，二维block中的线程ID计算：$id=x+yD_x$，其中$x$, $y$分别是两个维度的索引值，$D_x$表示x方向的维度。然后以连续的32个ID的thread作为一个warp。在设计kernel的时候需要考虑让一个warp中的thread访问连续的一片global memory。

shared memory分32个bank，每个bank是4字节，比如字节地址0\~3属于bank0，4\~7属于bank1，8\~11属于bank2……不同bank的数据可以同时被访问，同一bank的数据就不能一起访问。比如字节地址0和字节地址128的数据都属于bank0，就不能一起访问。

在矩阵乘法过程中，从global memory读数据，然后写入shared memory。不仅需要考虑从global memory的连续地址读取数据，而且在写入shared memory的时候也需要考虑减少bank冲突。

考虑到矩阵乘法axb的时候，a中取一行数据，和b中的一列数据计算内积。b本身就是按列存放数据的，warp中的线程也按照列来组织，这样能够保证warp中的线程访问的是shared memory中的连续的一片数据。但是要取a中的一行数据，就要跨行取数了，或者是当我从global memory读数据后，就将矩阵转置，再存到shared memory里。这里不管怎样都会涉及跨行的问题，而且如果block size是128字节的整数倍，那就肯定会有bank访问冲突。这里有个技巧就是在shared memory中额外多分配一点空间，从而让跨行的数据不再是128字节的整数倍，人为地让地址错开。我最后采用的做法就是将数据转置后存入shared memory，而在计算部分和的时候每个warp就可以从连续的地址加载数据了。在矩阵的维度足够大的时候，bank冲突是无法避免的，只不过通过这种方式能够充分利用访问shared memory的带宽，减少无效的数据访问。

```cpp
__global__ void coalesce(double *a, double *b, double *c) {
  // 数据按列存放，所以x按列方向增长，y按行方向增长
  int result_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int result_col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // 将结果清空
  c[result_col * N + result_row] = 0.0;

  // 每个block一起load数据，放入s_a s_b中，同一列的数据在地址上不能对齐，对齐的话会bank访问冲突
  __shared__ double s_a[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ double s_b[BLOCK_SIZE][BLOCK_SIZE + 1];

  // 每个thread需要load N/BLOCK_SIZE次数据
  for (int i = 0; i < N / BLOCK_SIZE; ++i) {
    // (y, x) thread 负责拷贝global中的 (y, x)的数据
    int s_a_row_in_global = result_row;
    int s_a_col_in_global = i * BLOCK_SIZE + threadIdx.y;
    int s_b_col_in_global = result_col;
    int s_b_row_in_global = i * BLOCK_SIZE + threadIdx.x;

    // s_a中的数据需要跨行存放，因为一个warp读的是列数据
    s_a[threadIdx.x][threadIdx.y] = a[s_a_row_in_global + s_a_col_in_global * N];
    s_b[threadIdx.y][threadIdx.x] = b[s_b_row_in_global + s_b_col_in_global * N];
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      c[result_col * N + result_row] += s_a[threadIdx.x][j] * s_b[threadIdx.y][j];
    }
    __syncthreads();
  }
}
```

从运行结果可以看出，在考虑上访存的过程后，同样维度的矩阵乘法进一步得到加速，而且与cublas实现的性能比较接近了。

![result](README/2024-04-16-23-02-47.png)

查看nsight systems的分析结果：

![nsight](README/2024-04-16-23-09-34.png)
