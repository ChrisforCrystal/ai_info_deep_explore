当在端侧设备（例如个人离线服务器）上部署离线AI应用（如Clawdbot这类助手）时，用户常常会直观地感受到响应速度偏慢。那么，如何有效加速你本地的“贾维斯”？

影响AI服务响应速度(TTFT)的因素很多，模型参数规模、能力以及部署硬件都是重要变量。但在相同的硬件与模型条件下，本地离线服务响应较慢的一个关键原因在于：它无法像云端那样利用大规模的分布式KV Cache。为突破这一限制，我们探索并实现了一种面向单机部署的推理加速方案，将KV Cache Swap技术应用于单台服务器环境。

Swap技术应用于推理端侧部署，可提升约30%的速度（基于vLLM近期实践）。该方案的主要思路是通过Swap特性提高Prefix Cache命中率，以针对性优化长序列场景中的Prefill性能，从而达到提升整体吞吐的目的。该方案在Prefix Cache命中率低时能做到零开销；同时，它完全独立于Scheduler模块的现有逻辑，侵入性低。特性PR已合入社区，本文将系统性地介绍该方案的原理与具体实现。

deepseek模型实测
实践背景：当前大模型的推理部署以云端形态为主。云端通常拥有大量服务器资源，因此适合部署如Mooncake、LMCache等分布式缓存系统。相比之下，端侧环境往往只有一至两台离线服务器，或仅是单一终端设备。在此背景下，面临的核心问题是：能否找到一种轻量、便捷的方案，在端侧设备（离线服务器、个人电脑）上也实现KV Cache的Swap功能？这也构成了本次实践探索的核心出发点。

1 Swap介绍
1.1 基本原理
为应对设备显存的限制，当前推理框架普遍在KV Cache管理中实现LRU淘汰机制，例如vLLM的Prefix Cache[1]与SGLang的RadixTree[2]。其原理是：当显存不足时，系统将据此淘汰那些最近使用最少的KV Cache。

radix attention淘汰数据
Swap指数据在不同存储介质间的移动，其核心价值在于扩展可用容量。应用例举：

通用场景：如虚拟内存，在主机内存与磁盘间交换数据，突破内存容量限制。
模型训练：指GPU显存与CPU内存间的Swap（CPU offload），用于解决训练显存不足，例如ZeRO[3]。
模型推理：用于维持更多KV Cache。可将被LRU淘汰的cache换出（Swap out）到其他介质，需用时再换入（Swap in）。

本文讨论的推理KV cache swap方式亦可称为KV cache offload或者CPU offload，目前尚未有一个统一的术语。当用“CPU offload”表示时，其与训练的“CPU offload”不是同一个概念，在此统一说明。

1.2 存储介质
KV Cache的存储体系一般可划分为四个层级：

GPU Memory：GPU的全局显存层，硬件基础为HBM。
Host Memory：主机（CPU）内存层，硬件基础为GDDR。
Local SSD：本地固态存储层，硬件为SSD或机械磁盘。
Shared Network Storage：共享网络存储层，涵盖分布式文件系统或云存储。
其整体构成一个性能与容量的金字塔：越靠近底层，存储容量越大，但访问延迟越高，传输带宽越低。

在Nvidia的Dynamo存储分类
1.3 KV cache传输路径
KV Cache的存储位置不同，对应的传输路径也不同，常见路径包括：

D2D：GPU → NVLink → GPU
H2D2H：远端CPU内存 → 本地GPU → NVLink → 目标GPU
H2D远端：远端CPU内存 → RDMA/IB/RoCE → 目标GPU
H2D本地：本地CPU内存 → PCIe → GPU HBM
D2H2D：远端GPU HBM → CPU内存 → 目标GPU
H2H2D：远端CPU内存 → 本地CPU内存 → GPU HBM
S2H2D远端：集群磁盘 → 本地/远端CPU内存 → GPU HBM
S2H2D本地：本地磁盘 → CPU内存 → GPU HBM
C2H2D：云端存储(S3/OBS) → TCP/IP → CPU内存 → GPU HBM

通道命名中：

H代表Host（即CPU主机内存）
D代表Device（即GPU）
S代表SSD（本地固态硬盘）
C代表Cloud（Shared Network Storage）
常见传输通道的典型带宽如下：

D2D NVLink：600GB/s或900GB/s；HCCS：~786GB/s
D2H/H2D：
PCIe 5.0 x16：~64GB/s
PCIe 6.0 x16：~128GB/s
RoCE（RDMA）：200GE或400GE
SSD：4~8GB/s
TCP/IP：12GE/25GE/50GE
1.4 KV cache优势分析
为了准确评估KV Cache带来的收益，首先必须理清一个关键问题：在何种条件下，KV Cache的传输会优于直接重计算？此前探讨过该问题（见[4]），根据实际场景，我们能对问题进行量化分析，衡量不同传输带宽对Attention计算时间的影响。

实验涉及的关键参数定义如下：

context length: 指代Prefill阶段需要处理的完整上下文长度。
cache_len: 指代已缓存并可复用的前缀（Prefix）长度。
seq_len: 指代当前需要执行Attention计算的新增序列长度。
有context_length = cache_len + seq_len，计算过程中：

仅考虑传输（Transfer）时，seq_len=1；
仅考虑重计算（Recompute）时，cache_len=0；
MLA模型参数：

# MLA的相关参数，参考DeepSeekV3参数

bs = 1
heads = 128
qk_head_dim = 128
kv_lora_rank = 512
h_dim = 7168
q_lora_rank = 1536
qk_rope_head_dim = 64
v_head_dim = 128
n_heads = v_head_dim
causal_mask_cof = 2 # casual mask是否启用 开启等于2，关闭等于1
n_shared_experts = 1
n_routed_experts = 256
moe_inter_dim = 2048
n_activated_experts = 8
若硬件的算力为400TFlops，0.5、1GB/S传输带宽下的计算结果。

当带宽超过 5GB/s 时，KV cache的传输优势将占据主导地位。

分析表明，传输优于重计算的阈值（即曲线交点）随模型参数与硬件配置动态变化。再看一个案例：

GQA模型参数：

# Qwen3 GQA参数：

bs = 1
heads = 64 # num_attention_heads
num_key_value_heads = 4
h_dim = 4096
head_dim = 128

基于H100、带宽10GB/s的测试环境中，对于GQA模型，传输方案的优势阈值对应的prefix cache命中长度约为9K。根据场景调整参数，能获得不同曲线。计算用例参考：kv_cache_transfer_vs_recomputation.ipynb[5]

由于KV Cache存储于Host Memory时，可通过PCIe通道获得超过30GB/s的有效带宽，该传输效率足以超越重计算。这一性能优势，为我们的方案实施提供了理论支撑。

2 方案设计
2.1 设计约束
功能界定：本方案专注于实现Prefix Cache在本地CPU内存层面的换入换出机制。多机环境下的KV Cache共享不在本次设计范围（与分布式KV cache的区别）。
集成规范：设计需遵循低侵入性原则，保持与现有Scheduler及Worker模块的逻辑解耦，避免对其代码进行修改。
2.2 vLLM框架相关逻辑介绍
下图展示了vLLM中V1版本KV connector的主要架构。该connector有两个执行角色：scheduler_connector和worker_connector，它们分别位于scheduler线程和worker线程中。两者通过元数据KVConnectorMetadata实现信息桥接：scheduler角色负责协调指挥KV数据传递，而worker角色则通过读取该metadata，精准确定需从远端加载的具体 KV 数据内容。

所有KV connector有个模板基类KVConnectorBase_V1（位置vllm/distributed/kv_transfer/kv_connector/v1/base.py)，这个模块涉及多个部分的实现，包括：

scheduler connector接口；
worker connector接口；
底层传输transfer layer接口；
2.3挑战与解决方案
挑战1：DP并行场景适配

在vLLM框架中，DP（数据并行）可跨多机部署，每个DP组拥有独立的Scheduler。这引出一个问题：多个DP组之间的CPU KV Cache应如何共享？

鉴于此案例的设计约束明确要求多机间不共享KV Cache，因此我们无需考虑跨机器DP组间的协同问题。然而，在同一台服务器内的不同DP组会访问相同的存储空间，即它们将共享同一个CPU KV Cache池。这需要解决随之而来的数据读写互斥与多进程通信问题。

解决方案：在vLLM实例上启动一个独立的Server，提供中心化的CPU KV Cache数据服务。每个DP组的Worker进程通过该Server访问共同的缓存数据（即Shared CPU Memory）。

实现要点

通信机制：使用ZMQ的ROUTER/DEALER[6]模式进行进程间通信。
内存结构：共享内存按(pp_rank,tp_rank)二维索引创建，结构为shared_memory[(pp_rank, tp_rank)]，以支持TP/PP并行。
启动时序：Server在所有Worker初始化完毕后，由Scheduler汇聚结果触发启动。
挑战2：消除Swap引入的开销

在方案原型验证阶段，我们发现在Prefix Cache命中率较低的场景下，会引入明显的额外开销，导致推理性能不升反降。根本原因在于：Prefix Cache会同步在CPU侧保存一份副本，而该保存操作完成前，相关进程会处于阻塞状态。

解决思路：采用异步Save与异步Load来消除此阻塞：

异步Save：请求处理结束后，仅将其放入一个待保存队列，随即进程立即返回，继续执行后续计算。
异步Load：当需要的数据命中Cache后，从共享存储中异步加载，不阻塞当前计算流。
方案1：KV cache逐层保存

其逻辑如下图所示。具体流程为：在模型每一层的计算完成后，立即触发一个异步任务来保存该层产生的KV Cache。计算操作与数据拷贝操作被安排在不同的CUDA(或者CANN)流中并发执行。

优势：能够快速地将KV Cache存储到CPU并生成对应哈希值，加速后续查询。
不足：此方案主要适用于PyTorch的Eager模式（非图模式），在图模式下可能受限。但考虑到PD分离场景中的Prefill阶段本就采用Eager模式，因此该方案在此场景下完全适用。

方案2：整体异步保存。

与“逐层保存”不同，此方案等待请求完整的forward计算全部结束后，再统一异步保存整个序列的KV Cache。

优势：该方式与计算图的融合度更高，因此能够良好地支持图模式执行，适用性更广。
不足：由于需要等待整个前向传播完成，KV Cache对应哈希值的生成时机存在滞后，这可能会略微增加后续请求的cache查询延迟。
以上两种方案的异步传输设计，均能确保即使在极端情况（如命中率为0）下，启用Swap特性也不会引入额外的性能开销。这一特性符合“vLLM的prefix cache为何零开销 [7]”所阐释的核心理念，即通过异步化与资源预占等手段，将潜在开销移出关键路径。

挑战3：Scheduler逻辑的兼容性适配

为遵循“不修改Scheduler原有逻辑”的约束，若仅依赖现有的KV Connector会遇到两个问题：

CPU KV Cache 无法在 Prefill 阶段被及时感知和利用。
被抢占后恢复执行的请求，其处理流程不会调用Connector。
这与Scheduler的现有逻辑直接相关：

哈希值使用：Scheduler在处理队列时，上一个请求的KV Cache哈希值可被下一个请求立刻复用。
Prefill阶段，GPUKVCacheManager和connector的使用顺序：

对于恢复执行的请求，Scheduler仅调用GPUKVCacheManager.allocate_slots()，而不会调用Connector。

解决方案：将CPU KV Cache哈希值的创建时机，延迟到请求处理完成之后。具体实现是：在Scheduler调用connector.request_finished()后的处理步骤中，再生成并注册可供后续请求复用的CPU KV Cache哈希值。

注：此方案暂未处理恢复请求（抢占后恢复）的场景。

2.4 整体方案
方案概述：我们在vLLM框架内，通过扩展KVConnector实现了CPUPrefixCacheConnector，专用于管理Prefix Cache向CPU内存的交换。

核心机制：本方案的Swap out数据并非被动淘汰KV cache，而是主动的冗余备份。具体来说，在请求的Forward过程中，所产生的KV Cache会同时存在于GPU显存和CPU内存中，形成双副本。该设计方式带来的优势是不用修改现有GPUKVCacheManager的逻辑，降低了开发工作量、减小了特性之间的耦合度。

关键模块：

CPUOffloadingConnector（交互层）：作为KV Connector的子类，它封装了与Scheduler和Worker模块的所有交互协议，是系统间的适配层。
CPUKVCacheMetadataProc（管理与调度层）：此模块扮演中心调度器的角色。其内部的CPUKVCacheManager类是核心，它参考了vLLM原有GPUKVCacheManager的设计理念与数据结构，并扩展了功能，以支持在单个节点内跨多个DP进程共享CPU端的KV Cache数据。

整体架构图
架构图中的Shared CPU Memory是一块共享内存区域，由CPUKVCacheMetadataProc创建并管理，供节点内所有DP组中的Worker进程共同访问使用。

2.5 基本步骤
运行逻辑分为两个部分，主要涉及四个模块的互动。

1、CPUKVCacheMetadataProc初始化。

2、KV Cache的查询、加载、保存。

为展示更细粒度的流程，可将CPUKVCacheMetadataProc（含MetadataServer、CPUKVCacheManager）和Worker（含ModelRunner、EngineCore）的内部组件展开描述。

3 代码与实测
3.1 代码介绍
本次开发主要实现了CPUOffloadingConnector和CPUKVCacheManager两大模块，并对模型文件进行了相应的适配修改。本特性的设计做到了与设备类型解耦，理论上可适配各种GPU硬件。但由于当前测试与验证环境皆基于PU，因此相关代码已合入vLLM的vllm-ascend仓库中[8]，PR:#1659。

3.1.1 metadata.py：此文件主要实现了MetadataServer、MetadataServerProc两个类。

其中，MetadataServer类负责管理KV缓存的全部元数据，其核心职责包括：初始化KV缓存、处理来自客户端的各类请求、以及分配与管理共享内存。在实现上，它使用ZeroMQ进行进程间通信（IPC），并利用共享内存来存储实际的KV缓存数据。

Metadata初始化时机

各WorkerConnector均会单独触发一次至Metadata的RPC调用，通过layer_spec参数获取其生成的共享内存索引。全体Worker初始化完毕后，SchedulerConnector继而调用Metadata的post_init()接口，宣告初始化完成。最终，由CPUKVCacheMetadataProc执行CPUKVCacheManager的初始化。

CPU shared memory：采用的是multiprocessing.shared_memory的SharedMemory。该过程会产生临时文件（/dev/shm），程序意外退出可能有残留，所以创建内存前会先清理上次可能残留的内存空间。

MetadataServerProc类中的run_metadata_server方法负责启动和运行元数据服务器，处理信号和异常，确保服务器的稳定运行。

3.1.2 cpu_kv_cache_manager.py：实现CPUKVCacheManager类，负责管理KV缓存的分配和释放，确保在分布式环境下正确处理KV缓存的生命周期。

3.1.3 cpu_offload_connector.py：继承自KVConnectorBase_V1，实现scheduler、worker相应的功能。

加载的触发逻辑：只有当命中长度超过一定阈值时才触发swap in逻辑，计算条件如下：

num_cpu_computed_tokens：CPU prefix cache命中长度；
num_computed_tokens：NPU prefix cache命中长度；
swap_in_threshold：设定加载阈值，为可配置参数。

加载过程：切换到对应的加载stream后，用非阻塞式的方式逐层加载。

保存方式：WorkerConnector在请求完成后，并不立即触发保存，而是将其插入save_input_queue队列并立即非阻塞返回。同时WorkerConnector会启动一个新线程来监听该队列：线程从队首获取请求并进行异步Save传输，仅当传输完成，才将请求标记为finish以供Metadata复用。

启动方式：在运行启动配置中添加：

--kv-transfer-config \
 '{
"kv_connector":"CPUOffloadingConnector",
"kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.cpu_offloading_connector",
"kv_role":"kv_both", "kv_connector_extra_config": {"swap_in_threshold": 0, "cpu_swap_space_gb": 800}
}'
其中，特性新增参数：

swap_in_threshold：命中率阈值；
cpu_swap_space_gb：CPU共享内存大小。
其它参数与KVConnector中的一致。

3.2 性能测试
测试方法：通过构造不同长度的前缀命中，控制GPU/CPU命中率，从而影响加载逻辑中的num_cpu_computed_tokens与num_computed_tokens。

观测目标：主要观测两者在不同取值下的结果差异，重点关注其差值diff。

实验条件:

测试模型：DeepSeek-R1-W8A8
测试机器：Ascend；
序列长度(total_tokens)：4k；
关键参数：--data-parallel-size 4 --tensor-parallel-size 4

TTFT对比（数值越小越好）
数据分析：

第一组数据：在命中率为0的情况下，测试了开启CPUoffload功能所带来的额外开销。观察到几乎无影响。
第二、三组数据：在保持GPU前缀命中率相同的条件下，逐步增加CPU的命中率，观察到首Token延迟（TTFT）显著下降。
第四、五组数据：持续提升GPU命中率，此时diff值保持稳定。相应地，纯GPU方案与GPU+CPU混合方案的TTFT差值变化也不大，说明在此区间性能提升趋于平缓。
除上述对照实验外，我们还测试了实际业务场景的收益：在缓存高命中（即diff值较大）时，性能收益可超过50%。这表明，本特性在重复前缀较多的应用场景中尤为适用。

3.3 方案总结
优势：

部署简便：特性随vLLM服务一同启动，无需额外部署独立的插件或工具，实现了开箱即用。
零开销设计：在缓存命中率较低的场景下，该特性不会引入额外的性能开销。
场景适配性好：尤其适用于端侧部署环境，且在包含大量重复前缀的推理任务中，能够发挥出显著的性能优势。
当前不足：缓存生效延迟，由于采用异步保存机制，新生成的CPU KV Cache需要经过数个推理轮次后才能被后续请求查询并使用，存在一定的延迟。

待探索内容：

图模式兼容性：在开启图模式（Graph Mode）时，若采用逐层加载策略，其中插入的CPU操作会破坏计算图的连续性，此问题有待进一步研究与解决。
共享内存实现限制：理论上，CPU共享内存可通过pin*memory及torch.tensor.share_memory*()实现。但在当前实测中，通过ZMQ与Pickle进行跨进程传输时，不同进程获得的张量内存地址不一致的情况，这导致该方案暂不具备落地可行性，需在后续研究中予以解决。
保存去冗余：为了避免修改KV Manager逻辑，KV cache保存采用的是双冗余机制。但对于长期存在于GPU中的KV cache不必要拷贝到Host memory中。保存机制待进一步优化。
从技术原理可知，本方案的实现与硬件设备类型无关，可适用于NPU、GPU及其他类型芯片。由于PyTorch对CUDA的支持更为成熟，在GPU上进行实践所能获得的性能收益通常会更明显。感兴趣的同学可参考本实例对应的PR，自行进行适配与验证。

文中理论计算代码地址：

https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/kv_cache_transfer_vs_recomputation.ipynb
