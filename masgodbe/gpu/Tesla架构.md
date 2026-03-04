引言：Tesla架构可以认为是第一代真正开始用于并行运算的GPU架构，其硬件设计与细节奠定了当前并行运算架构的基本形态， 尽管主流的显卡架构Hopper(2022年3月发布，H100)的算力已经远超Tesla，但其运作模式、单元细节的设计理念基本保持不变，所以认识Tesla 架构，是了解GPU显卡硬件的基础。 本文通过分析Tesla架构的第一代（G80）和第二代（GT200），帮助读者对Tesla硬件有个基本认识。

读前问题：

Tesla架构的前后，GPU的发展是什么样的？
Tesla架构两代的性能提升了多少？
图形卡与并行运算卡的结构差异？
Tesla架构支持一些什么样的CUDA特性操作？
运算、内存、指令单元的工作时钟频率是否相同？
底层计算处理单元的工作机理是怎么样的？
warp的双指令工作的原理
1 发展历史简介
NVIDIA GPU从第一代GPU（1999年）GeForce 256图形显卡，到如今的Hopper（2022年）数据运算卡。显卡主要系列发展成了以GeForce（图形/游戏卡）、Quadro（渲染卡）、Tesla（数据卡）为主的三个方向，其中Tesla系列的初代架构Tesla架构也是这些显卡的发展基础。在Tesla架构之前的显卡主要是图形显卡(GeForce）产品，之后衍生出了强大的并行计算卡。Tesla架构本身也经历两代的发展：G80系列和GT200系列。

1.1 Tesla系列并行计算卡的发展
Tesla系列随着架构的演进，算力成倍的增长，从数据运算卡（Datacenter Products）的路线看一下架构的演进。Tesla系列是从Tesla架构开始的，如图展示第二代tesla 到volta的发展：

Tesla -&gt; Volta
后面再从Volta->Hopper（NVIDIA后面发布会没有给出相应系列比较图片）又经过了几代的演进，总体来看，架构发展的时间（注：数字代表显卡算力）：

1：Tesla（2008年）-> 2：Fermi （2010年）-> 3：Kepler（2012年）-> 4：？->

5：Maxwell（2014年）-> 6：Passcal（2016年）-> 7：Volta（2018年）->

7.5 Turing（2018年）-> 8：Ampere（2020年）-> 9：Hopper（2022年）。

数据中心具体产品的算力：

Datacenter 算力列表
1.2 Tesla架构之前的图形显卡的发展
Tesla 架构之前的显卡也经历了几代的发展，但基本上是图形显卡。从GeForce1到GeForce7，下面按照时间发展简单介绍：

GeForce （代号NV10，1999年）是由NVIDIA研发的第五代[显示核心]。此核心常简称为GeForce，这亦是NVIDIA第一个以"GeForce"为名的显示核心。NVIDIA于1999年8月发布。

GeForce 2（代号为NV15， 2000年），是由NVIDIA设计的第二代GeForce显示核心，于2000年4月26日推出。GeForce 2 GTS（NV15）是整个家族第一款显卡。GTS代表GigaTexture Shader。它的像素填充率达到每秒16亿。

GeForce3（代号是NV20，2001年）是NVIDIA发明的第三代显示核心。它是全球第一款支持DirectX8的显示芯片。

GeForce4 （核心代号为NV25，2002年）是NVIDIA研发的第4代绘图处理器。架构实际基于GeForce3改进而成。

GeForce 5（官方统称为GeForce FX系列，2004年）分为两大系列：GeForce FX（核心代号NV3x）在2002年11月18日的COMDEX展上发布；GeForce PCX（核心代号NV3xPCX）在2004年2月发布。

GeForce 6 （代号为NV40， 2004年）。支持微软的DirectX9.0c规格下，全数均支持Vertex及Pixel shader 3.0版。NVIDIA用了10亿美金研发。该系列于2004年4月14日推出。

GeForce 7 （2005年）是第七代绘图处理器，3D引擎升级为CineFX 4.0。全系列绘图晶片是原生PCI-E。与GeForce 6一样，支持DirectX 9.0c，Shader Model 3.0, OpenGL 2.0，HDR。

GeForce 8，代号G80，是第八代GeForce显示芯片。在7900 GTX发布后八个月，NVIDIA于2006年11月推出GeForce 8800 GTX，它是基于G80核心。

Tesla 架构是在G80（第一代）的基础上发展起来的，后面经过两年改进，推出了Tesla二代：200系列。Tesla 架构的讲解主要围绕G80的G8800和GT200的GTX280进行分析。

2 Tesla 硬件结构主体
Tesla系列的初代架构Tesla在2006年就应用到了显卡G80系列上面，其架构关键特点是推出了NVIDIA第一代“统一着色与计算架构”(unified shader and compute architecture)，经过改进又推出了第二代统一着色与计算架构 GeForce 200系列，在该系列中有图形架构和计算架构两个版本， 后面并行计算卡发展成了DataCenter产品系列，图形处理也单独的衍生出两个分支GeForce和Quadro产品。 总体趋势是衍生出了许多专用化程度更高的产品，不止限于图形处理功能（虽然大家依然习惯称这些产品为GPU（Graphic Process Unit））。

2.1 主体架构
G80架构图(Tesla一代)

G8800架构
第一代Tesla是一种图像与计算一体式的GPU架构，如图所示，主机（host cpu）和系统内存（system memory）通过内部BUS总线+接口（host interface，一般是PCI/PCIE插槽）与GPU交互。整个架构从上到下大致分为三个区域：

调度与分发区域：包括顶点任务分发通道：Input assembler + Vertex work distribution； 像素任务分发通道：Viewport/clip/setup/raster/zcull + Pixel work distribution; 计算任务分发通道 Compute work distribution。这些通道负责具体的计算任务的准备以及匹配对应的计算单元来下发相应的任务。
计算区域：主要有阵列式的处理族TPC（ texture/processor cluster）组成，TPC负责完成具体的运算工作，TPC的数量可根据需求改变，。
存储与处理区域：主要完成存储和一些预处，包括：光栅操作器ROP（ raster operation processor）、缓存L2 cache、全局内存DRAM。
GTX280架构图（Tesla二代图形版本）

Tesla二代图形处理的架构图与G80架构基本类似，主要不同的地方在于，他TPC单元SM数量是3个。调度分发区域的功能模块有所差异，但基本上都是在完成Geometry、Pixel、Vertex、Compute等任务的预处理与调度工作。

GTX280架构图（Tesla二代并行计算版本）

并行计算版本架构图的主要特点是调度单元的功能更加单一，由一个任务编排器（Thread Scheduler）组成。同时去掉了ROP单元，增加了一个原子操作Atomic单元，原子操作能够降低并行数据的管理开销，提高数据处理速度。

2.2 硬件参数
产品参数：

8800有多款GPU产品：其算力差距也相差比较小：

GeForce 8 系列参数
en.wikipedia.org/wiki/GeForce_8_series
其中GFLOPS理论达到了416（小于芯片的peak值576 GFLOPS），在当时已经是相当高了（虽然最新的GPU已达到500TFLOPS）。

200系列的高端卡参数：

GeForce 200系列参数
en.wikipedia.org/wiki/GeForce_200_series
GFLOPS理论值基本在1000左右。相比G80系列翻了一倍。

参数比较

Tesla 一代和二代，在整体上面的都做了改进，具体可以通过GeForce 8800 GTX 和 GeForce GTX 280做一下比较：

GeForce 8800 GTX vs GeForce GTX 280
运算特性支持（Feature Support） ：

G80算力（Compute Capability）基本上都是1.0和1.1，具体参数如下所示：

G80 系列显卡算力
en.wikipedia.org/wiki/GeForce_8_series
算力对应了CUDA支持的功能如下：

CUDA特性与算了对应关系
en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
主要是增加了原子操作（Atomic functions）是一种thread（CUDA里面的线程）级别写安全的操作，thread之间操作时不会出现数据误操作，类似于c++线程的原子操作。算力1.1 上面增加两个对全局内存（global memory）上面的原子操作：32位整形、32位浮点数的atomicExch操作。

> Integer atomic functions operating on 32-bit words in global memory
> atomicExch() operating on 32-bit floating point values in global memory
> 注：atomicExch操作是一个交换操作，原型：int atomicExch(int\* address, int val); 将address位置的值返回，并把val值作为新值赋给address指向的值。

1.2算力的GPU在全局内存上面增加了一个64位整形的原子操作，同时在共享内存（shared memory）增加了32位整形、32位浮点数atomicExch的原子操作。而且还多了一个在warp里面的warp vote functions操作，由于需要理解warp，所以在讲完warp的概念后再阐述（看第4节）；

> Integer atomic functions operating on 64-bit words in global memory
> atomicExch() operating on 32-bit floating point values in shared memory
> Integer atomic functions operating on 32-bit words in shared memory
> Wrap vote functions
> 到GT200出现了CUDA1.3算力特性(GT200a/b GPUs only)，是双精度浮点操作（Double-precision floating-point operations ）。相应地，GT200系列里面在SM里面增加了一个直接支持浮点运算的硬件单元（FMAD）。

注：由于年代久远，目前CUDA官网上面查算力特性的时候，显示的都是3.5以后的特性：

2.3 芯片基板
G80基板（G8800）

首先看一下芯片（GeForce 8800 Ultra）的基板，主要的模块件是：ROP/memory x 2、Texture x 2、SMs x 4、Setup/raster/dispatch、HostPCIE； 其中“Setup/raster/dispatch” 包含了之前提到的的分发（distribution）模块功能。

G8800 die layout
整个基板的电气参数如下：

681百万晶体管，面积
工艺90nm
128 SP核 ，16个SM
处理时钟频率1.5Ghz
支持768M GDDR3 DRAM
384引脚的DRAM通道，1.08GHz 时钟频率
104GB/S 峰值带宽。
功率150W 1.3V
值得注意的是基板是不包括DRAM的，像DRAM（以后的HBM）存储是属于片下存储单元，通过DRAM 引脚与芯片链接。

GT200基板

得益于当时的65nm工艺，在GT200系列的基板相比于G80的基板整体布局做得更加的紧凑，面积不增大多少的情况下电气参数提升了不少，放一张GTX200基板的图片：

该芯片系列相比于上一代，主要是在工艺上面得到改进：

14亿晶体管，面积
工艺65nm
240 SP核，30个SM
3 模块功能介绍
3.1 调度分配单元
调度分配单元，主要是完成数据进入TPC之前的处理工作，这个单元基本上会根据不同的显卡的特性进行设计调整，所以不同规格的GPU名称会有所差异。比如在G80中叫做任务分发器（work distribution）、GT200称之为全局模块调度器（global block scheduler），总体来说它主要完成任务命令的读取，然后根据任务分发命令到具体TPC单元的SMs中执行。调度分配单元里面的细微的工作又可以由具体的功能模块完成，下面例举G8800和GT200的调度单元

3.1.1 G80的调度分配单元：

G80的调度分配单元的结构链路比较多，根据2.1中的架构图可知，主要包括顶点任务分发通道，像素任务分发通道、计算任务分发通道。这些通道运行时还包括了命令的获取与数据的传递工作。

顶点（vertex）任务分发通道：主要完成顶点着色以及几何着色（geometry shader）。顶点着色的处理结果传递到一些图像处理的功能模块（Viewport/clip/setup/raster/zcull ）最后得到像素块（pixel fragments）。

像素（pixel）任务分发通道：像素任务处理将像素块通过TPC进行作色处理，最后把输出传递到ROP进行光栅处理形成输出。

计算（compute）任务分发通道：将需要计算的线程传递到合适的TPC中。

3.1.2 GT200(并行计算架构)的调度分配单元:

GT200并行计算架构的调度分配单元，在架构上来说相对清晰，如下图所示（GTX280），只有一个global block scheduler（或者Thread Scheduler）。这个单元的调度策略是“回合机制”（round-robin），主要是在计算任务线程块（thread blocks）中找到合适的SMs执行任务。

注：在调度分配单元认识中需要注意一个细节，就是这些单元的运行时钟clock 与SMs、以及显存DRAM的clock是相互独立的，所以可能同时存在三个不同工作的时钟频率。

3.2 TPC单元
TPC（texture/processor cluster or thread processing cluster）单元是一个运算的基本功能块，多个TPC组成了SPA（streaming processor array (SPA) ），TPC的模块图如下所示：

两者对比可以看到TPC 的组成有些共同单元，还有各自特殊单元，其中主要的部分是SM、Textrue(Cache)、SMC（SM controller）。在一个TPC中SM共用L1cache，SM受SMC单元的控制。

3.3 SM单元
SM（streaming multiprocessor）单元是指令执行的最小单元。它主要是根据操作指令使得线程在对应的硬件上面完成具体的工作。

3.3.1 G80的SM

如图所示主要包括：指令缓存（I chache）、常量缓存（C cache）、指令获取与下发单元（MT issue）、运算核（SP，streaming processor )、特殊运算单元SFUs （special function units） 、共享内存（shared memory）。

3.3.2 GT200的SM

如图所示，相比G80 它主要多了一个64位的FMAD （fused multiply-add )单元（已用红框标出）。

GT200SM
对这里面的一些单元分别说明：

加乘单元（SP core/ CUDA core）：MAD(multiply-add units)能够进行加ALU（integer）和乘FPU （floating point）运算。 主要负责完成32位的warp指令运算，一次运算需要4个时钟周期（clock cycles）。
SFU里面的 FMUL（floating point multiply units）：完成普通的乘法运算。运算时消耗4个时钟周期。
SFU: 可以完成sine、cosine、平方根等数学操作，运算消耗16个时钟周期。
Branch：条件控制运算单元，完成条件控制指令。运算消耗4个时钟周期。
寄存器：存储直接操作数；
共享内存: 线程之间的共享数据存储；
常量内存：保存一些线程之间能够快速获取的相同的变量
对于显存存储这一块的知识，可以参看上一篇写的文章：

GPU内存(显存)的理解与基本使用
1099 赞同 · 63 评论 文章
3.3.3 DUAL issue

CPU执行任务一般是消耗1个时钟周期，而GPU的SP运算单元/FMUL运算/Branch运算，却需要消耗4个时钟周期，但是GPU里面的SM的线程执行指令（warp instruction）下发只需要两个时钟周期，时间是执行单元的1/2。为了让两个不同类型的运算单元同时工作，Tesla的SM 设计出了一种双指令（dual issue）机制，有些单元能够并行。比如 SP单元能够与 SFU同时工作，如下图所示：

warp指令分发（ISSUE）单元给 SP单元里面的FPU下发MAD操作，下一个指令给SFU下发MUL乘法操作，这样使得：指令1和3给了SP单元连续执行；指令2和4给了SFU单元，也能够连续执行。

注：指令之间是否可以用dual issue，要看硬件支持程度，Tesla架构支持的相对较少，比如64位的FMAD就需要独立运行。

3.4 SM的工作机理

SM的工作的机制是SIMT(Single-instruction, multiple thread)，即单个指令控制着多个thread工作。

那么SM多线程如何组织起来？---> 通过Warp为单位来完成，warp看成thread的一个组织者，一个warp大小为32 \* thread。 SM管理着一个由多个warp组成的池子，比如G80的单个SM 有24个warps， 即单个SM最多管理着
个threads，warp与warp之间按照时间顺序依次执行，如图所示：

注意：

1、warp的排序是由SM scheduler决定。不是一个简单的FIFO队列，还考虑到了匹配与开销等问题。

2、SIMT 与SIMD（Single-instruction, multiple data）有着较大区别。SIMD只是说数据。SIMT的范围更广，因为Thread里面不止只有数据。

3.5 Texture和数据的读写
Texture：纹理单元（Texture）是一种针对GPU图像应用场景设计的存储处理单元，texture指令的操作输入一般是纹理坐标，而输出可以是像RGBA这样的颜色坐标，整个过程完成了坐标到过滤后的样本的转换。

一般而言，在cpu上面存储数据都是线性且连续的，而且读取数据往往是按照某一个数量单位（比如64B）读到缓存cache中，要读取小于这个数量单位的数据时，会浪费掉大量带宽，比如读取局部的图像数据（大小4B）信息时，会读入60B无用的信息。为了解决这个问题，texture单元设计成2维坐标（现在支持3D多维功能）拿取并处理数据的能力，texture被设计为只读属性，且操作顺序相对严格。所以：
texture的带宽开销更小，时延更低，但操作灵活度更小。

在硬件上面，texture有配套的地址生成器单元address generation units (AGUs)和过滤器单元（filter units）。像G80的8800Ultra显卡的单个纹理单元包括AGUs x 4 和 filter units x 8。其处理的能力是：38.4 giga bilerps/s [ bilerp（a bilinear interpolation of four samples, eg RGBA color）]

看一下texture和读写时的数据链路对比如下图所示（GT200）：

首先，数据的保存和加载都是由SM下发执行，但通道逻辑有所差异，且保存和加载数据使用不同的时钟。

数据加载与存储的过程：首先，SM控制器下发的数据查询指令（warp指令）的执行，这个过程需要进行地址计算，完成虚拟地址到物理地址的映射，然后在通过内部总线（intra-chip corssbar bus）将信息传递到内存控制器（Memory controller）然后读取数据。存储的与加载类似，数据存储时可能经过ROP单元（如果有ROP单元）。

虽然单个内存操作动作是由一个warp指令来完成，但是内存控制器每次只能服务半个warp 即16个thread的内存操作！

texture数据路径：

全局内存（RDMA）-> 内存控制器 -> 纹理缓存（texture cache） -> AGUs -> SM控制器 -> 具体的SM单元
注意：数据加载(LOAD)和texture的数据共用一个物理路径，所以两者操作是互斥地。

ROP单元：ROPs (raster operation processors) 定函数光栅操作处理器，在memory上直接处理颜色和深度/模板等。）。

4 知识补充
Warp vote functions
Tesla 二代出现了的vote操作，能够极大的提高threads之间的数据处理速度。warp的vote操作，可以理解是对warp里面的thread之间的信息进行比较然后进行操作，涉及的CUDA函数包括，

**any,**all, 和\_\_ballot。

int \_\_any(int predicate); warp里面只要有一个thread里面传入的predicate非零，则返回1，否则返回0；

int \_\_all(int predicate); warp里面所有的thread里面传入的predicate非零，则返回1，否则返回0；

unsigned \_\_ballot(int predicate)；warp里面的N个thread里面传入predicate非零，则返回的数的第N位置位（设置为1）。

any和all 比较好理解，类似python语言里面的对一个数组进行操作，然后结果由数组共同决定，比如：

any([1, 2, 0, 0, 0]) = True
\_\_ballot，通过如下示意图来解释其操作过程，假设Warp 一共有8个thread 编号0~7，每个thread 的predicate值如右边所示，最后返回的result的二进制位由thread上面的predicate决定，非零即置位。最后的结果为0b10110110。

里面的predicate 可以设置为条件比较，比如 x > 0。在CUDA9.0之后这些函数都被修改成如下格式：

        int __all_sync(unsigned mask, int predicate);
        int __any_sync(unsigned mask, int predicate);
        unsigned __ballot_sync(unsigned mask, int predicate);
        unsigned __activemask();

主要是多了一个mask（掩模）参数，通过掩模操作可以控制参与vote的线束，比如mask的值为0b00000001（十六进制0x01），表示第0个thread参与计算。如果全部生效（non-exited），则设置为0xFFFFFFFF。

<注：文章若有错误/不足，欢迎评论指正（私信@kaiyuan）。致力于帮助更多的人学习与进步>

---
