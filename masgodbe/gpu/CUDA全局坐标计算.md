在CUDA编程和代码阅读过程中，计算偏移坐标（offset）/全局Idx是一项频繁遇到的任务。这种计算至关重要，因为它建立了线性数据结构与并行计算线程之间的对应关系。CUDA通过三维的线程组织结构，实现了这种映射。线程的组织和管理涉及到几个核心参数：gridDim、blockDim、blockIdx和threadIdx。

本文将主要解释CUDA线程的组织方式以及偏移坐标的计算方法。包括：在使用三维网格（3D grid）和三维块（3D block）的线程结构时，如何进行全局坐标计算。以及解释，数据指针 \*src 传递给CUDA内核（kernel）后，线程如何根据预先定义的顺序，准确地读取指向的数据。

threads相关概念
坐标计算公式
示例代码 （代码位置）
1 threads相关概念
CUDA里面用Grid和Block作为线程组织的组织单位，一个Grid可包含了N个Block，一个Block包含N个thread。

相关单位参数：

gridDim: blocks在grid里面的数量维度；dim3;
blockDim: threads在一个block的数量维度；dim3；
blockIdx: block在grid里面的索引；dim3；
threadIdx：thread在block里面的索引;dim3;
gridDim，规定blocks的形状，blockDim规定了threads的形状。 这些参数在kernel写好后，**global**/**\_device**定义的函数里面能够直接访问到变量。我们在创建kernel时，通常会传入blocks的维度、threads维度，示例如下

**global** void kernel(float \*src){
//do something
}

// 定义grid 和block的形状：
dim3 BlocksPerGrid(N, N, N); // gridDim 对应gridDim.x、gridDim.y、gridDim.z
dim3 threadsPerBlock(M, M, M); // blockDim 对应blockDim.x、blockDim.y、blockDim.z
// invoke code:
kernel<<<BlocksPerGrid, threadsPerBlock>>>(\*src);
下图给出了一个实例，其中gridDim .x=2, .y=2, .z=3; blockDim .x=4, .y=2, .z=4

这样，一共包含的block数量：2*2*3 = 12；每个block线程总数：4*2*4 = 32； 线程总数：12 \* 32 = 384

一个线程组织结构示例
若要索引一个threads，可通过索引坐标, 如图标记为蓝色的线程，其索引表示方式：blockIdx的 x=1, y=0, z=2; threadIdx的x=3, y=0; z=3

而线程的全局idx的求解，还需要按照坐标计算公式求得：

2 坐标计算公式
现在需要让线程通过全局索引拿取各自的数据，则需要通过公式转换计算求得。首先需要知道全部3D的坐标下的idx如何计算。3D3D的索引计算即找出一个线程在所有线程中的位置。由于线程的组织结构包含了两层，所以可以拆分计算：

step1: 计算线程thread在block的位置：
step2: 计算该block在grid中的位置；
step3: 计算block有多少线程；求解位置索引。
计算公式例举中，src为源数据是1维结构（对应了数据指针）。

step1 公式：

其中，threadIdx表示是线程的索引，blockDim代表了block的尺寸。上例中若单独只看block部分（如下图所示）， blockDim(.x=4, .y=2, .z=4)，threadIdx的x=3, y=0; z=3, 那么该线程thread在block的位置threadInBlock的结果：

step2 公式：

计算该block在grid中的位置；需要知道blockIdx和 gridDim。 还是上面的示例，已知gridDim .x=2, .y=2, .z=3; blockIdx的 x=1, y=0, z=2; 即可得

step3 求解全局的idx：

2.1 全3D结构：3Dgrid-3Dblocks
**global** void kernel3D3D(float *input, int dataNum)
{
// thread在block中位置计算：
int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// block在整个grid中的位置计算：
int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
// 一个block有多少个线程计算：
int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
// 位置索引：
int idx = threadInBlock + oneBlockSize*blockInGrid;
}
接着，根据3D公式，将其中不需要的维度设置为1，不需要用到索引设置为0，既能获取其它不同维度的公式，去处维度的顺序一般是先Z、再Y。

2.2 全2D结构：2Dgrid-2Dblocks
令 threadIdx.z = 0; blockIdx.z = 0; blockDim.z = 1; gridDim.z = 1; 带入3D公式中简化得到2D计算：

**global** void kernel2D2D(float *input, int dataNum)
{
// int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
// int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
// int idx = threadInBlock + oneBlockSize*blockInGrid;
// when:
// threadIdx.z = 0; blockIdx.z = 0;
// blockDim.z = 1; gridDim.z = 1;
// then:
// int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x;
// int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x;
// int oneBlockSize = blockDim.x*blockDim.y;
int idx = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);
// thread overflow offset = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
}
2.3 全1D结构：1Dgrid-1Dblocks
令 threadIdx.y = 0; threadIdx.z = 0; blockIdx.y= 0; blockIdx.z = 0; blockDim.y = 1; blockDim.z = 1; gridDim.y = 1; gridDim.z = 1; 带入3D公式中简化得到1D计算：

**global** void kernel1D1D(float *input, int dataNum)
{
// int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
// int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
// int idx = threadInBlock + oneBlockSize*blockInGrid;
// when:
// threadIdx.y = 0; threadIdx.z = 0; blockIdx.y= 0; blockIdx.z = 0;
// blockDim.y = 1; blockDim.z = 1; gridDim.y = 1; gridDim.z = 1;
// then:
// int threadInBlock = threadIdx.x;
// int blockInGrid = blockIdx.x;
// int oneBlockSize = blockDim.x;
int idx = threadIdx.x + blockIdx.x * blockDim.x;
// thread overflow offset = blockDim.x*gridDim.x;
}
2.4 其它结构
可根据自己需要（一般是数据结构）构建出自己需要的thread结构形式。方法：需要什么维度时保留对应计算维度，不需要的维度设置为1，不需要用到索引设置为0，通过3D3D公式即可得到对应idx计算方式。

2Dgrid-3Dblocks，grid中的z维度不需要用到，所以设置：blockIdx.z=0; gridDim.z=1；

**global** void kernel2D3D(float *input, int dataNum)
{
// int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
// int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
// int idx = threadInBlock + oneBlockSize*blockInGrid;
// when
// blockIdx.z = 0;
// then
int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x;
int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
int idx = threadInBlock + oneBlockSize\*blockInGrid;

}
3Dgrid-2Dblocks，block中的z维度不需要用到，所以设置：: threadIdx.z=0; blockDim.z=1;

**global** void kernel2D3D(float *input, int dataNum)
{
// int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
// int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
// int oneBlockSize = blockDim.x*blockDim.y*blockDim.z;
// int idx = threadInBlock + oneBlockSize*blockInGrid;
// when
// threadIdx.z=0; blockDim.z=1;
// then
int threadInBlock = threadIdx.x + threadIdx.y*blockDim.x;
int blockInGrid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
int oneBlockSize = blockDim.x*blockDim.y;
int idx = threadInBlock + oneBlockSize*blockInGrid;

}
类似的还有1Dgrid-2Dblocks（矩阵计算中常用）、1Dgrid-3Dblocks、3Dgrid-1Dblocks、1Dgrid-2Dblocks，方法相同在此不赘述。

3 示例代码
3.1 打印线程中的全局索引
可以将把线程的索引全部打印出来，检验是否出错，比如打印一个2D2D的线程结构

**global** void printIdx2D2D()
{
int i = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);
printf("Global idx %d, threadIdx.x: %d, threadIdx.y: %d threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d \n",\
 i, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}
用grid 3 x3 block 2 x 2 可知，最大的thread数量为 36 （3x3x2x2）通过如下调用方式：

printIdx2D2D<<<dim3(3, 3), dim3(2,2)>>>();
打印如下所示， 可以看到Global idx （0~35）每个线程相互独立，各不相同：

3.2 构建一个测试用例
计算目标：用CUDA线程对数据进行自增1的运算。 通过cpu运算来校验计算过程是否有误。

假设坐标的映射关系求解错误，可能会导致数据出现重复的运算（或者漏运算），则最终结果也是错误的；若映射关系正确，则GPU求解结果与CPU结果应当保持一致。通过一个test来验证结果，代码逻辑如下：

#define TOTAL_SIZE 5000
#define N 4
#define M 4
using kernel = void (_)(float _, int);

bool test(kernel func, dim3 BlocksPerGrid, dim3 threadsPerBlock) {
unsigned int totalSize = TOTAL_SIZE;
float* hostData = (float*) malloc(sizeof(float) _ totalSize); // 主机数据
float_ checkData = (float*) malloc(sizeof(float) * totalSize); // 校验数据
float* devicePtr;
checkCudaErrors(cudaMalloc((void\*\*)&devicePtr, sizeof(float) * totalSize));
for (int i =0; i < totalSize; ++i) {
hostData[i] = i;
checkData[i] = i + 1; // 校验数据增加1
}
checkCudaErrors(cudaMemcpy(devicePtr, hostData, totalSize _ sizeof(float), cudaMemcpyHostToDevice));
func<<<BlocksPerGrid, threadsPerBlock>>>(devicePtr, totalSize); // 通过GPU进行运算
checkCudaErrors(cudaMemcpy(hostData, devicePtr, totalSize _ sizeof(float), cudaMemcpyDeviceToHost));
// check result: 此处校验结果
bool rst = true;
for (int i =0; i < totalSize; ++i) {
if (!areFloatsEqual(checkData[i], hostData[i])) {
rst = false;
printf("The result not equal in data index %d. expect:%f result:%f\n", i, checkData[i], hostData[i]);
break;
}
}
checkCudaErrors(cudaFree (devicePtr));
free(hostData);
free(checkData);
return rst;
}
源码位置：threads_hierarchy_calc

编译与运行：

nvcc -lcuda threads_hierarchy_calc.cu -o test && ./test
