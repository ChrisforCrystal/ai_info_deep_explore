深度解析FlashMLA: 一文读懂大模型加速新利器
kaiyuan
kaiyuan​
大模型话题下的优秀答主
已关注
收录于 · 分布式并行训练
194 人赞同了该文章
​
目录
收起
1 计算原理分析
1.1 计算公式
1.2 Attention分块运算
2 计算流程分析
3 关键代码解析
3.1 Warp group的建立
3.2 数据双缓冲
3.3 KV分页管理逻辑
3.4 SM负载均衡
3.5 测试用例
FlashMLA是一种在变长序列场景下的加速版MLA（Multi-Head Latent Attention），针对decoding阶段优化。开源地址：FlashMLA，其主关键内容：

性能（H800 SXM5 CUDA12.8）：3000 GB/s HBM带宽；580 TFLOPS吞吐

特点：

存算优化：双warp group计算流设计与应用（存算重叠、数据双缓存）；
分页缓存：KV page block管理提升显存利用率，更好适应变长序列；
SM负载均衡：动态调整block数据，充分利用GPU算力。

FlashMLA分块逻辑
当前代码仓的限制条件：

针对NVIDIA Hopper架构GPU优化
数据类型：BF16/FP16
分页缓存管理粒度：64-block
仅包含推理场景
1 计算原理分析
1.1 计算公式
FlashMLA这个库解决的是什么计算问题?

MLA计算主要包含升/降秩线性计算和attention计算部分，FlashMLA完成MLA中MHA计算部分，不负责升/降秩的线性乘法操作。MLA的结构如下图所示：

https://arxiv.org/pdf/2405.04434
计算MLA的MHA和通常的MHA计算存在差异，先来分析如下问题：

问题1：FlashMLA的计算公式输入结构组成是什么样的？

一般而言，计算MHA时需要Q/K/V三个输入值，而MLA由于引入升降秩操作，算MHA时输入值发生了变化。 MLA的公式如下，FlashMLA完成（公式46）的计算。

在deepseekV2中有提到矩阵W可以调整，具体是
转移到
计算中，
转移到
计算中。

公式46进行一下调整：

输入的参数变为
、
，然后根据参数配置计算输入的维度。

# deepseekV3，671B config：

"kv_lora_rank": 512,
"qk_nope_head_dim": 128,
"qk_rope_head_dim": 64,
"v_head_dim": 128,
输入的Q/K Head_dim：为
组合：kv_lora_rank + qk_rope_head_dim = 576；

输入的V Head_dim：为
的head dim值：kv_lora_rank =512。

注：这里就解释了为什么head_dim 不是config里面的128值。
1.2 Attention分块运算
输入、输出明确后需要对KQV进行分块计算（按照FlashAttention类型原理）， FlashMLA的分块逻辑如下：

大致步骤：

从Q取q_block单位，从K取k_block单位完成qk运算'、softmax运算得到p_block;

从V取v_block单位，然后分块成两份，分别与p_block计算得到o_block0和o_block1刷新到结果O上；

外层循环（outer loop）：每次加载一个q_block；
内层循环（inner loop）：每次加载一个kv_block;
其中分块运算，使用两个不同warp group完成。

动图封面
单个inner loop的示意动图
2 计算流程分析
怎么利用Hopper架构提速分块MLA的计算过程，使其达到“Flash”的标准？

回答这个问题需要先了解一下hopper架构的一个例子（cutlass库）：Ping-Pong计算方式。

技术上称为"sm90_gemm_tma_warpspecialized_pingpong"，采用异步流水线运行，利用 warp 专精化。与更经典的同构内核不同，"warp 组"承担专门的角色。请注意，一个 warp 组由 4 个 warp 组成，每个 warp 有 32 个线程，总共 128 个线程。 Warp是GPU中的基本执行单元,由32个线程组成，Warp group是多个warp的集合用于协同工作（理解warp原理参看：GPU硬件分析的第三节）
该操作采用生产者（Producer）、消费者（Consumer）模式。Cutlass的Ping-Pong例子中包含1个生产者、2个消费者，如下图所示，生产者专门负责搬运数据，消费者完成计算。采用这种模式能够更充分的利用TensorCore。

Ping-Pong的流水线对硬件调用如下图所示，涉及关键模块：计算warp组、访存warp组，SMEM、GMEM、TMA存储单元和TesnorCore计算单元。

操作步骤参看：https://pytorch.ac.cn/blog/cutlass-ping-pong-gemm-kernel/
RMEM(register Memory)：寄存器；SMEM(Shared Memory)：共享内存；GMEM（Global Memory）：全局内存HBM
TMA（Tensor Memory Accelerator）：TMA 允许在 GPU 的全局内存（Global Memory）和共享内存（Shared Memory）之间异步传输
有了对ping-pong方法的理解后，阅读FlashMLA的计算逻辑就更容易了。FlashMLA里面也使用了两个warp组。与ping-pong不同的是，这两个warp组是一种协作关系。

warp_group_0(线程0~127): 负责进行attention scores运算、mask、softmax、部分PV计算。
warp_group_1(线程127~255)：负责加载数据Q和K、部分PV计算。
根据代码（flash_fwd_mla_kernel.h)，其计算逻辑主要在compute_attn_1rowblock_splitkv_mla函数中，这里对其过程进行分析如下所示：

该kernel包含了一个row（一次外层循环迭代）的完整运算，其主要步骤：

warp_group_1从GMEM加载数据q和第一个k_block数据到SMEM；每个kernel只加载一个q_block（outer loop）； 在内层循环（inner loop）中每次迭代开始前加载一个k_block； block的大小为：64(kBlockM) × 576(kHeadDim)
warp_group进行计算前有个线程同步，保证数据加载完成/运算结束；
warp_group_0调用gemm完成attention scores获得QK值，然后进行mask和softmax计算，得到P值；
warp_group_0将P值copy一份到共享内存中，接着warp_group_0和1进行一次同步， 确保copy完成后。warp_group_1从共享内存中加载一份P值；
warp_group_0和warp_group_1调用gemm完成 PV计算，分别获得O结果的一部分（两者合起来O值：64\* 512）
warp_group_0把softmax计算的row_sum以及row_max存入共享内存，两个warp组同步后，warp_group_1从共享内存获得该值。
KV内层循环完成一个O的刷新，Q外层循环完成整个结果刷新。
把这些步骤结合模块运算

加载数据Q/KV

通过QK计算获得P值

warp_group0的PV计算

warp_group1的PV计算
包含两层循环的完整的运算流程：

动图封面
运算流程
3 关键代码解析
FlashMLA的代码整体结构比较清晰，主要计算在flash_fwd_mla_kernel.h中实现。

主函数：

template<typename T, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
static_assert(Headdim == 576);
FLASH_ASSERT(params.d_v == 512);
FLASH_ASSERT(params.k_ptr == params.v_ptr); // Shared_KV
using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, 512>;
run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
}
从run_flash_splitkv_fwd_mla为主开始执行，包含两个kernel，对应attention分块运算的两个步骤：

1 flash_fwd_splitkv_mla_kernel：

处理原始输入(Q、K、V) 执行分块的注意力计算
通过一个for循环遍历数据，通过compute_attn_1rowblock_splitkv_mla运算获得一个row单位的结果(64\*512)
2 flash_fwd_splitkv_mla_combine_kernel：

处理中间结果(oaccum、lseaccum)合并分块结果
FlashMLA的外层循环在flash_fwd_splitkv_mla_kernel中实现，

内层循环在kernel：compute_attn_1rowblock_splitkv_mla中实现。

3.1 Warp group的建立
Warp group分组大小是多少，代码中如何区分不同的group？

Flash Attention MLA的实现中，Warp group的划分是通过Kernel traits中的参数隐式设置的。 在Flash_fwd_kernel_traits_mla结构体中定义了关键参数：

template<int kHeadDim*, int kBlockM*, int kBlockN*, int kNWarps*, typename elem*type=cutlass::bfloat16_t, int kHeadDimV* = 0>
struct Flash*fwd_kernel_traits_mla {
// 总的warps数量
static constexpr int kNWarps = kNWarps*; // 通常设置为8
static constexpr int kNThreads = kNWarps _ 32; // 8 _ 32 = 256个线程

    // 用于softmax计算的warps数量
    static constexpr int kNWarpsS = 4;  // 4个warps组成第一个group
    static constexpr int kNThreadsS = kNWarpsS * 32;  // 4 * 32 = 128个线程

warp group的单位是128个线程，区分warp_group的idx可调用cutlass里面API完成：

canonical_warp_group_idx函数的大概逻辑：

int canonical_warp_group_idx() {
// threadIdx.x是当前线程在block中的索引
// 32是每个warp的线程数
return threadIdx.x / (32 \* WarpsPerGroup);
3.2 数据双缓冲
K/V_block的加载为什么需要双缓冲区？

为了方便warp_group_0和warp_group_1的计算通信掩盖，在计算中k的SEME buffer大小设置了两份。

单个buffer的弊端：假设每次只加载一份K/V_block，那么在进行gemm_qk->gemm_pv运算时，是无法加载新的K/V_block的，数据覆盖会导致GEMM的计算结果出错，就需要进行同步操作。

同步操作会降低效率所以设置K/V双缓冲区，当一个buffer运算时，另一buffer加载数据，实现计算和数据访问操作重叠。

时间轴：----------------------------------------------------------->
K/V缓冲区1：[加载Block1][计算Block1][加载Block3][计算Block3]...
K/V缓冲区2： [加载Block2][计算Block2][加载Block4]...
看一下双缓冲区的代码逻辑：

// 双缓冲区的设置
cute::array_aligned<typename Kernel_traits::Element,
cute::cosize_v<typename Kernel_traits::SmemLayoutK> \* 2> smem_k; // Double buffer

// 在循环中的使用
if (n_block % 2 == 1) { // 初始设置
constexpr int sK_offset = size(sK);
tSrK.data() = tSrK.data() + sK_offset / 8; // 指向第二个缓冲区
tOrVt.data() = tOrVt.data() + sK_offset / 8;
}

// 在主循环中
for (; n_block >= n_block_min; --n_block) {
flash::cp_async_wait<0>(); // 等待前一次的数据复制完成
\_\_syncthreads();

    if (n_block - 1 >= n_block_min) {
        const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK); // 在两个缓冲区之间切换
        tKsK.data() = tKsK.data() + sK_offset;

        // 开始异步加载下一个block的数据
        flash::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
            gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK);
    }

    // 使用当前缓冲区的数据进行计算
    // ...

}
3.3 KV分页管理逻辑
在seq维度上，FlashMLA进行了序列维度的KV分页缓存管理，将长序列分成多个64token的blocks。分页缓存的基本逻辑如下，通过分块+索引表来完成数据存储/访问。

原始序列：[seq1: 157长度] [seq2: 89长度] [seq3: 213长度]
分块后： [3个block] [2个block] [4个block]
(64+64+29) (64+25) (64+64+64+21)

物理内存布局：
[B1_1][B1_2][B1_3][B2_1][B2_2][B3_1][B3_2][B3_3][B3_4]

Block表：
seq1: [0, 1, 2] // 指向物理页0,1,2
seq2: [3, 4] // 指向物理页3,4
seq3: [5, 6, 7, 8] // 指向物理页5,6,7,8
优势：可以将数据分散到物理内存中。

可以利用碎片显存（减少碎片）
sequence的长度变化处理更加灵活
以分块组织方便更新和替换
// flash_fwd_mla_metadata.cu
void build_block_tables(int *block_tables, // 输出的block表
int *num_splits, // 每个batch需要的split数量
const int *cu_seqlens_k, // 每个batch中K的累积序列长度
const int *seqlens_k, // 每个batch中K的序列长度
int batch_size, // batch大小
int page_block_size) { // block大小(64)

    // 对每个batch
    for (int i = 0; i < batch_size; i++) {
        const int seqlen = seqlens_k[i];  // 这个batch中K的序列长度
        // 计算需要多少个block来存储这个序列
        const int num_blocks = DIVIDE_ROUND_UP(seqlen, page_block_size);

        // 对这个序列的每个block
        for (int j = 0; j < num_blocks; j++) {
            // 直接使用连续的block索引
            block_tables[i * max_num_blocks + j] = j;
        }
    }

}
3.4 SM负载均衡
FlashMLA中实现了一个简单的SM计算单元的动态负载均衡逻辑（flash_fwd_mla_metadata.cu），目的是让每个SM单元均衡的拿到KV-page block，过程：

计算总工作量、计算每个SM的目标工作量:
// 计算每个序列的block数量并累加总量
int total_num_blocks = 0;
for (int i = threadIdx.x; i < batch_size; i += 32) {
int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
total_num_blocks += num_blocks + fixed_overhead_num_blocks;
num_blocks_shared[i] = num_blocks;
}
....
// 将总工作量平均分配给每个SM
int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks; 2. 动态分配:

while (now_idx < batch_size) {
int num_blocks = num_blocks_shared[now_idx];
int now_remain_blocks = num_blocks - now_block;

    if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
        // 如果剩余负载足够处理整个序列
        cum_num_splits += now_n_split_idx + 1;
        num_splits_shared[now_idx + 1] = cum_num_splits;
        remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
        ++now_idx;
        now_block = 0;
        now_n_split_idx = 0;
    } else {
        // 如果剩余负载不足，需要分割序列
        if (remain_payload - fixed_overhead_num_blocks > 0) {
            now_block += remain_payload - fixed_overhead_num_blocks;
            ++now_n_split_idx;
            remain_payload = 0;
        }
        break;
    }

}
3.5 测试用例
测试用例遍历不同参数下的性能，一个测试执行后先打印参数：

b=128, s_q=1, mean_sk=4096, h_q=16, h_kv=1, d=576, dv=512, causal=True, varlen=False
其中：

b: batch size (128)
s_q: query序列长度 (1或2)
mean_sk: key序列的平均长度 (4096或8192)
h_q: query的head数量 (16, 32, 64, 128)
h_kv: key/value的head数量 (固定为1)
d: 每个head的维度 (576)
dv: value的维度 (512)
causal: 是否使用因果mask (True)
varlen: 是否使用可变长度序列 (True或False)
然后输出测试结果：

0.193 ms, 476 TFLOPS, 2760 GB/s
计算的公式代码：

FLOPS = s_q _ total_seqlens _ h_q _ (d + dv) _ 2
bytes = (total_seqlens _ h_kv _ d + b _ s_q _ h_q _ d + b _ s_q _ h_q _ dv) \* ( torch.finfo(q.dtype).bits // 8)
BW = bytes / 10 \*\* 6 / t
其中FLOPS的计算是只考虑kv_scores、qkv 的gemm运算，且不考虑causal_mask ，参考deepseek3mfu中计算公式：

kv_scores = (2 _ gbs _ seq_len² _ num_heads _ qk_head_dim) / (causal_mask ? 2 : 1)
qkv = (2 _ gbs _ seq_len² _ num_heads _ v_head_dim) / (causal_mask ? 2 : 1)
FLOPS = 2 _ s_q _ total_seqlens _ h_q _ d _ 2 + 2 _ s_q _ total_seqlens _ h_q \* d

片上通信带宽计算考虑：输入Q、KV数据搬运到芯片上，以及输出O数据写回GMEM。

数据总量 = Q + KV + O = b _ s_q _ h_q _ d + total_seqlens _ h_kv _ d + b _ s_q _ h_q _ dv

想深耕AI Infra领域？欢迎访问InfraTech库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套50+知识干货及适合初学者的notebook练习。
