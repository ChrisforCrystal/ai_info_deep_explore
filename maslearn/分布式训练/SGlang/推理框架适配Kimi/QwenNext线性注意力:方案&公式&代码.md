在推理框架(vLLM/SGLang)上适配混合模型，关键是构建对应的线性模块。适配的过程不仅是搭建一个新的模型层，还要考虑适配框架的现有逻辑，涉及的问题包括：KV cache的管理、算式的构建、并行策略的适配等，本文以QwenNext、KimiLinear模型的GDN/KDA进行简单分析，帮助读者快速了解相关知识。

1 模块架构简介
目前效果较好的混合模型中，线性注意力（Linear Attention）模块与标准注意力（Standard Attention）模块会按照一定比例（如3:1）混合。标准注意力模块一般采用GQA/MLA；线性模块像QwenNext模型使用的是GDN(GatedDeltaNetworks)，KimiLinear用的是KDA(Kimi Delta Attention），两个模块基本结构类似，都是基于Delta规则(Delta Rule)的线性注意力的改进。GDN和KDA主要的差异点：

算式上S的衰减项的处理不一样；
两个模块映射计算、激活函数处理有区别。

模块架构如上图所示，主要运算包括线性映射、Conv、L2、alpha、beta、注意力算式等。进一步了解GDN/KDA模块中各个参数的含义，推荐阅读：

LLM线性注意力(LinearAttention)的原理与细节(AlphaDeltaGate)解析
150 赞同 · 2 评论 文章
2 KV cache的管理
一般来说KV cache仅用于Attention，在vLLM或者SGLang里面增加新模块不必修改KV cache的管理。在GDN和KDA的线性模块中也没有KV cache，但这类模块中却引入了跟请求相关的需要被cache的内容：

递推式的状态值：用S(SSM)表示;
短卷积的输入数据：用Conv表示

与KV cache不同的是，S&Conv不会随序列的变长而递增，对单个请求而言是固定大小。但是对于整个推理框架而言，S&Conv会随请求的数量变化而变化，所以必须考虑多请求S&Conv的管理。

在原基础上适配新类型的cache，涉及的问题包括：

KV cache的逻辑管理是保持不变，还是新增类型？
预留多少空间存储S&Conv，数据排列如何设计？
PD分离中数据传递时，如果开启了TP并行，数据该如何打包？
2.1 当前逻辑
以vLLM框架为例，先简要看一下标准attention的KV cache管理方案。

vLLM基于paged attention原理的KV cache管理分为两层：物理层和逻辑层。

逻辑层中，由Scheduler通过KV Manager管理着KV Pool逻辑块，通过页表（block table）完成逻辑与物理层映射。

物理层中，KV Cache按照layer来创建的，所有请求共用页（page）。常见的模块是MQA和MLA，虽然其页的分配存在差异，但block使用方式相同。

2.2 线性注意力cache管理
设计原则上，线性注意力的cache管理尽量与框架现有逻辑保持统一，避免设计一套新的管理逻辑增加维护成本。

块的分配结构：S&Conv不需要像KV cache一样得构建链表（block list）来管理块，主要是数据排布，考虑的实现方式：

a) 层与层之间独立，对于单个请求对应的每一层分配一个固定的块；

b) 请求与请求之间独立，单个请求所有线性注意力层的块在一起，占用一个连续片段。

参数的大小计算，当获取到整体可用显存（available memory）后，标准attention的每一层能创建多少的blocks计算方式：

blocks = available_memory // page_size // num_layers

page_size = block_size _ heads _ head_dim \* dtype

block中包含的tokens为固定值。当使用混合注意力后，标准attention的available_memory 值要先减去线性attention层的显存开销。这个值可以根据用户设置的max_num_seqs以及层参数算得。

如果仅考虑数值计算，最后线性attention和标准attention的page_size可能会大小不一致。为了便于内存管理/提升计算效率，统一page_size的大小，要增加对显存的动作：

step1：调整KV blocks数量，让其与S&Conv基本相等；
step2：在较小值的尾部增加pad对齐大小；

2.3 cache的存储大小对比
当序列多长时，线性层的cache才有访存/传输优势？ 可以通过计算来寻找答案。

GDN/KDA的cache计算。单层S&Conv的存储大小与模型的配置参数相关，FP16/BF16数据，计算方式：

S:= 2 _ num_v_heads / TP _ head_k_dim \* head_v_dim

Conv:= 2 _ (head_k_dim _ num_k_heads _ 2 + head_v_dim _ num_v_heads)/TP \* (conv_kernel_size - 1)

注意：S&Conv计算没有考虑投机（speculative）推理，若开启投机推理，还要加上对应的长度。

MLA/GQA的cache计算。在MLA/GQA中 kv cache的存储大小与序列长度相关，计算方式：

MLA:= 2 _ seq_len _ (kv_lora_rank + qk_rope_head_dim)

GQA:= 2 _ 2 _ seq_len _ num_key_value_heads _ head_dim / TP

通过上述的计算方式，结合模型的参数。可以计算出分界点序列长度，举两个例子（假设TP=1）：

例1：Qwen3-Next-80B-A3B-Instruct，模型相关参数：

# GDN:

"linear_num_key_heads": 16,
"linear_num_value_heads": 32,
"linear_value_head_dim": 128,
"linear_key_head_dim": 128,
"linear_conv_kernel_dim": 4,

# GQA:

"num_key_value_heads": 2,
"head_dim": 256,
通过逐步增加序列长度，得到曲线如下，交点处的tokens=536。

例2：Kimi-Linear-48B-A3B-Instruct，模型相关参数：

# KDA:

    "head_dim": 128,
     "num_heads": 32,
    "short_conv_kernel_size": 4

# MLA:

     "qk_rope_head_dim": 64,
     "kv_lora_rank": 512

通过逐步增加序列长度，得到曲线如下，交点处的tokens=975。

结论：当序列长度超过一个交汇位置tokens数量时，线性注意力层才有cache的优势。

这个计算主要是考虑对访存、以及cache传输的影响，总的显存对比的考虑模型整体情况。

2.4 相关特性的调整
cache的传输：线性注意力层传输与KV cache的传输方式保持一致。考虑的关键点是：线性attention有多头且开启了TP功能时，TP域内不同设备存储的S&Conv不一样，而MLA的KV cache是冗余存储，在传输的时候需要分情况处理。

Prefix cache：由于线性注意力层保留的是最后的状态，所以一般而言prefix cache匹配只能匹配最长序列。如果要匹配中间的结果，得让线性注意力层保存对应的S&Conv的中间状态数据。考虑的实现方案：

离散保存：按照block为单位进行状态保存，例如128个tokens保存一次S&Conv。这种方式下存储cache呈现分段增长。
以算换存：存储GDD/KDA的输入值hidden states，或者映射后的输入值（k/v/alpha/beta），通过重算的方式恢复S；
与Prefix cache有着类似问题的特性还有投机推理。

并行策略：GDN和KDA模块投影层可用列切+行切的方式开启TP并行，如下图所示。结合chunkwise运算基本满足长序列运算需求。由于是按照head切分，所以cache的管理逻辑不需要调整。

TP并行
线性层一般不必开启SP/CP并行，若序列在GDN/KDA计算前被切分，将其数据聚合(gather)即可。由于没有softmax计算，如果要支持CP并行，实现上并不复杂。采用GDN/KDA的并行式计算，主要的问题是cache管理中每个设备上的S&Conv存储是否需要冗余。

目前在vLLM和SGLang中的S&Conv管理都是沿用的Mamba模型的逻辑。最近社区都有在完善在优化这块的功能，比如vLLM中将Attention的page size进行大小对齐，如下图所示，MambaSpec（S&Conv存储）和AttentionSpec（KV cache存储）page size大小相等（适配模型Nemotron-Nano-12B-v2）。

在Tensor的读取方面，调整了Mamba的排布方式保证两种cache做到兼容，更加方便加速算子（FlashAttention）的处理。目前方案适配到了vLLM的V1版本，至于V0版本中Mamba cache已被弃用。

3 算式的构建
在GDN和KDA的推理中会用到两种算式形态：递推和分块(chunkwise)算式。对于推理的prefill而言，输入序列比较短时用递推算式运算，当序列较长用分块方式效率更高；而decode阶段，一般用递推公式计算即可。下面以GDN为例进行计算构建分析。

3.1 递推算式
GDN状态S更新的计算公式：

输出的计算公式：

为了得到状态值和输出结果，先要完成输入值的处理。

映射计算Q/K/V、G（支持多头，每个头计算相同）计算公式：

其中核函数
用Silu，Conv为短卷积运算，这里的G是gate控制输出（不是gamma衰减）。

# 公式4使用的卷积运算：

conv1d = nn.Conv1d(
in_channels=self.conv_dim,
out_channels=self.conv_dim,
bias=False,
kernel_size=self.conv_kernel_size,
groups=self.conv_dim,
padding=self.conv_kernel_size - 1,
)

# 公式5的处理：

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
inv_norm = torch.rsqrt((x _ x).sum(dim=dim, keepdim=True) + eps)
return x _ inv_norm
alpha的计算公式：

其中A和
(dt_bias)为可学习参数，推理阶段直接从权重中加载数值。

# Pseudo code

A = torch.empty(num_v_heads)
A_log = nn.Parameter(torch.log(A))
dt_bias = nn.Parameter(torch.ones(num_v_heads))

# A_log、dt_bias 需要从权重里面加载

# Gamma值的运算(公式7)：

g = -A_log.float().exp() \* F.softplus(a.float() + dt_bias)
beta的计算公式：

公式1的S拆分成两步：

# Pseudo code

# 迭代计算过程：

out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim)
state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)
for i in range(sequence_length):
q_t = query[:, :, i]
k_t = key[:, :, i]
v_t = value[:, :, i]
g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
beta_t = beta[:, :, i].unsqueeze(-1)
kv_mem = (state _ k_t.unsqueeze(-1)).sum(dim=-2) # 公式10
v_new_t = v_t - kv_mem # 公式9
state = state _ g_t + k_t.unsqueeze(-1) _ (v_new_t _ beta_t).unsqueeze(-2) # 公式2
out[:, :, i] = (state \* q_t.unsqueeze(-1)).sum(dim=-2)
3.2 Chunkwise算式
UT转换公式：

其中M表示mask运算，
表示严格下三角运算（StrictTril）在torch中直接调用tril(-1)即可。
表示gamma衰减矩阵，定义满足:

# Pseudo code

    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    # chunk衰减计算：
    g = g.cumsum(dim=-1)
    gamma = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril()).exp()

    # 公式11:
    T = -(K_beta @ K.t() * gamma).tril(-1)
    for i in range(1, C):
        T[i, :i] = T[i, :i] + (T[i, :, None] * T[:, :i]).sum(-2)
    T += torch.eye(C)
    # 公式12:
    W = T @ (K_beta * g.exp().unsqueeze(-1))
    # 公式13：
    U = T @ V_beta

对输入累乘得到
，一般给函数传入的值是未经过e指数运算的
值，计算上面可以先进行累加后算指数。对于gamma的除法运算的处理可以先用减法运算，再进行指数运算。很显然的是前者计算效率更高。

g.cumsum(dim=-1).exp() 等价于g.exp().cumprod()；
(g1 - g2).exp() 等价于 g1.exp() / g2.exp()；
状态刷新与结果输出：

# Pseudo code

def chunk_gated_delta_rule_forward(Q, K, V, beta, g, C):
"""
Q/K/V: query, key, value of shape [L, d]
beta: beta of shape [L]
g: gate of shape [L], with exp()
C: chunk size
""" # L: sequence length, d: head dimension
L, d = Q.shape # 分块：
Q, K, V = map(lambda x: x.reshape(-1, C, d), [Q, K, V])
beta = beta.reshape(-1, C)
g = g.reshape(-1, C)
K_beta = K _ beta.unsqueeze(-1)
V_beta = V _ beta.unsqueeze(-1)

    # chunk衰减计算：
    g = g.cumsum(dim=-1)
    gamma = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril()).exp()

    # 公式11:
    T = -(K_beta @ K.t() * gamma).tril(-1)
    for i in range(1, C):
        T[i, :i] = T[i, :i] + (T[i, :, None] * T[:, :i]).sum(-2)
    T += torch.eye(C)
    # 公式12:
    W = T @ (K_beta * g.exp().unsqueeze(-1))
    # 公式13：
    U = T @ V_beta
    S = torch.zeros(d, d)
    O = torch.zeros_like(U)
    for i in range(L//C):
        q_i, k_i, u_i, w_i = Q[i], K[i], U[i],  W[i]
        # 公式15:
        new_v_i = u_i - w_i @ S
        o_inter = (q_i * g[i].exp()) @ S
        o_intra = (q_i @ k_i.t() * gamma[:, i]).tril()
        # 公式16:
        S = S * g[i, :].unsqueeze(-1) \
            + (k_i * (g[i].unsqueeze(-1) - g[:, i]).exp()).t() @ new_v_i
        # 公式17:
        O[i] = o_inter + o_intra @ new_v_i
    return O.reshape(L, d)

4 框架中代码的实现
目前现状：GDN和KDA在vLLM/SGLang的适配沿用了Mamba模型的已有逻辑，主要工作是将cache管理稍作调整，并实现对应的加速算子。在功能方面，框架中已有相关代码实现；而性能方面，开源代码还在优化迭代中。

下面以SGLang的KDA为例来进行介绍。

4.1 算子实现
线性注意力模块里面多个步骤都可以用融合算子来提升效率，所以Mamba模型中Conv运算、递推式均有对应的融合算子，GDN和KDA在算子实现方面也都有对应的融合算子实现。

递推算式：在GND的基础上，KDA修改了对于alpha运算部分。FIA(flash-linear-attention)库中能够找到GND Triton版本的实现，目前增加KDA逻辑分支：

# 代码位置：sglang/python/sglang/srt/layers/attention/fla/fused_recurrent.py

# 函数：def fused_recurrent_gated_delta_rule_fwd_kernel

# 增加 IS_KDA=True, 逻辑调整：

# ...

if not IS_KDA:
p_g = g + bos _ HV + i_hv
else:
p_gk = g + (bos _ HV + i_hv) \* K + o_k

# ...

        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])

# ...

        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K

# 函数入口

# 代码位置：sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py

# L 386

initial_state = ssm_states[cache_indices].contiguous()
fused_recurrent_kda(
q=q,
k=k,
v=v,
g=g,
beta=beta,
initial_state=initial_state,
use_qk_l2norm_in_kernel=True,
cu_seqlens=query_start_loc,
)
其中cache获取方式：

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal

Chunkwise算式，输入处理与前面一致，主要是算子与算子的调用存在差异：

# 函数入口

# 代码位置：sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py

# L 399

        initial_state = ssm_states[cache_indices].contiguous()
        (
            core_attn_out,
            last_recurrent_state,
        ) = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )

# 算子实现

# 代码位置：sglang/python/sglang/srt/layers/attention/fla/kda.py

# 关键函数接口：chunk_kda_fwd

# L1150

def chunk_kda_fwd(
q: torch.Tensor,
k: torch.Tensor,
v: torch.Tensor,
g: torch.Tensor,
beta: torch.Tensor,
scale: float,
initial_state: torch.Tensor,
output_final_state: bool,
cu_seqlens: torch.LongTensor | None = None,
)
4.2 cache管理的适配
SGLang中为Mamba类模型定义了一个cache的管理的池子（Pool），KimiLinear在此基础上进行了修改：

# 代码位置：sglang/python/sglang/srt/mem_cache/memory_pool.py

# L 124 class MambaPool

# 增加分支判断：

self.is_kda_cache = isinstance(cache_params, KimiLinearCacheParams)

# ... 部分逻辑：

            if self.is_kda_cache:
                conv_state = [
                    torch.zeros(
                        size=(num_mamba_layers, size + 1) + conv_shape,
                        dtype=conv_dtype,
                        device=device,
                    )
                    for conv_shape in conv_state_shape
                ]

# ...

4.3 当前相关的PR
Kimi Linear

主要PR：

vLLM: https://github.com/vllm-project/vllm/pull/27809
SGLang：https://github.com/sgl-project/sglang/pull/12469
Qwen Next

Qwen Next实现在vLLM/SGLang里面的与之类似，也继承了Mamba的现有逻辑。主要PR：

SGLang：https://github.com/sgl-project/sglang/pull/10233
vLLM：https://github.com/vllm-project/vllm/pull/24526
几个关键文件的代码位置：

模型：vllm/vllm/model_executor/models/qwen3_next.py
算子：vllm/vllm/model_executor/layers/fla/ops/fused_recurrent.py
mamba基类：vllm/vllm/model_executor/layers/mamba/abstract.py
值得注意的是，在开源框架中模型的性能可能低于官方的数据，一方面，模型厂商一般有自己的内源版本，迭代速度更快所以内容领先开源版本；另一方面，使用的硬件有可能不是NVIDIA的GPU，而目前开源的内容基本针对的是GPU。若要优化开源线性模块的性能，可以考虑结合硬件特性，从算子多流、图模式等方面着手
