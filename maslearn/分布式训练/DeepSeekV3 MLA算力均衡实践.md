在LLM的推理性能优化探索中，输入的波动变化会导致模型不同模块对算力的需求发生改变。如何应对这种变化，使硬件算力发挥到极致？是性能优化研究的重点。目前已有一些系统的解决方案，例如动态负载均衡、专家负载均衡和模型切分。最近我们尝试了一种模型层面的计算均衡方案，该方案适用于MLA的Prefill阶段，具有改动量小、均衡性好的优点，可实现20%~50%+性能提升，相关PR已开源。

1 问题模型
目前MoE并行部署中常采用多个attention+单个FFN形态。模型的Attenion计算阶段用DP>1，DP域之间运算独立；而FFN计算阶段采用大EP并行（DP域打通），不同GPU拥有不同的独立专家。

这种部署方式带来的问题是：计算速度较快的 DP Attention 模块会出现空转等待。这是因为，推理请求的计算量及 KV Cache 大小与其输入和输出序列的长度正相关。当不同请求的序列长度存在差异时，就会导致各个 DP 实例间的计算速度产生快慢不均。（在分布式推理优化思路2.2中有讨论此问题)

序列变化对Attention计算的影响

在MLA的计算中，输入序列的变化对prefill与decode影响不相同，以单请求为例进行讨论。根据MLA的flops计算公式（参考：超细图解MLA计算流&吸收矩阵对比分析），可得到MLA的计算量随着序列的增长的变化数据，也能分析各个操作的占比。

采用DSV3配置参数，得到MLA中Q/K/V/O线性层、attention层的算量与序列长度的关系曲线。

Prefill

Prefill阶段计算量（开启causal mask）
根据数据可知，在prefill阶段：

attention的计算复杂度与序列长度呈二次方关系，而其余部分的计算量则与序列长度呈线性关系。
当序列较短时，线性层（o_linear）的计算占主导；随着序列增长，attention的计算占比逐渐增大。
需要注意的是，当输入长度小于4K时，Attention的计算占比通常低于50%。
Decode

在 decode 阶段，性能主要受限于 IO 带宽，因此本文暂不讨论该阶段下 MLA 的算力不均衡问题，而将重点放在 prefill 阶段的计算分析上。在 MLA 内部，可按计算过程是否包含 head 维度，将其分为两类：

包含头(has head dim)计算：attention(scaled dot-product)、q_up_proj、kv_up_proj

不包含头(non head dim)计算：q_down_proj、kv_down_proj、out_proj

从图中可以看出以下两点：

i. 相同序列长度下两类计算的占比变化：当序列长度大于约4K时，分头计算部分的占比更高。这表明，在不同长度的序列场景下，算力均衡优化的侧重点应有所区别。

ii. 不同序列长度间的计算量对比：序列越长，总计算量越大，且增长趋势呈非线性。为直观感受这一差距，我们对比长度为20和2K的序列，二者总计算量之比为：

flops ratio = 20_len_total_flops / 2000_len_total_flops ~= 0.0082

较短的请求所需的 MLA 算力不足长序列的 1%，这种显著差异最终会导致执行时间出现不均。

2 方案设计
2.1 初步思路
为了解决长、短请求给到不同DP域产生的负载不均衡问题，我们在实践当中，对DP之间的请求进行优化。在PD分离场景下，考虑对MLA的并行运行方式进行调整：

a) 对于不涉及 head 维度的计算，采用序列混合并均匀分配的方式进行优化；

b) 对于涉及 heads 的计算，则打通 DP 域，使所有 rank 各自计算部分 head，且每个 rank 所处理的 head 互不重复。

这相当于以跨 DP 域的 SP-TP-SP 方式对 MLA 计算过程进行分解。该方法在CloudMatrix384[1]技术报告有分享，其思路类似于Megatron3[2]中SPTP混合并行。

首先回顾 Megatron3 中 DP 域内的 SP-TP 方案：线性层采用序列并行（SP），Attention 模块则使用 head 维度的张量并行（TP）。在序列并行中，单个 rank 所处理的序列长度为seq_len / SP；在张量并行中，单个 rank 所拥有的 head 数量为heads / TP。

Megatron3 所讨论的是同一 DP 域内的并行方案（每个 DP 域包含一个完整的 Attention 模块）。若要将此方案扩展为跨 DP 域的 SP-TP-SP 并行，关键在于解决跨域协作问题。

解决方式：在 SP 阶段进行序列重组。通过跨 DP 域混合序列并均匀分配，使每个 DP 域获得长度基本相等的请求。下图展示的是两个序列进行混合 SP 的示例：

在 TP 阶段，同一份模型参数会在不同 DP 域之间按 head 维度进行均分。首先通过 allgather 操作使每个 DP 组获取所有请求的序列数据，随后各 DP 组在内部独立计算所分配到的部分 head。下图为一个简单示意图：

这里需要特别注意 DP 域的概念发生了变化。在传统定义中，单个 DP 域通常包含一个完整的模型副本（replica）；而在上述方案中，每个 DP 域仅负责模型层的部分计算。也可以理解为此时 DP=1，即未开启传统的模型数据并行。

通过结合跨 DP 域的 SP 与 TP 方式，我们将 MLA 的计算过程划分为三个阶段进行处理。每个阶段都涉及序列的重新调整，因此引入了两次额外的集群通信。下图展示了包含 4 个请求的处理流程：

stage1和stage3都采用混合SP，包括操作：down_proj和o_proj；
stage2采用TP，包括操作：q_up_proj, kv_up_proj, FA(scaled dot-product)。

SPTP的示例图
上述的方案SP按序列均分、TP按头均分，使得计算达到了token细粒度的负载均衡。

2.2 二次调整
在实施过程中，尽管2.1方案有效解决了负载均衡问题，但在MLA的具体实现中仍面临以下几项挑战：

问题1：如何处理额外添加的通信（allgather + alltoall）造成的影响？

解决措施：在down_proj与up_proj之间引入的跨DP域的allgather，考虑让它与MoE阶段 TP并行产生的allgather结合。alltoall则暂不处理，因为原本o_proj做TP切分需要一个reduce scatter操作，替换成alltoall的影响可忽略。

挑战1：框架的 KV Cache 管理逻辑应如何适配？

在 prefill 阶段，涉及两个关键的 KV Cache 管理行为：一是将 KV Cache 块分配给请求，二是通过前缀缓存（prefix cache）将匹配的 KV Cache 输出提供给激活值。

面对挑战1，为了保证KV cache的管理逻辑不变，只修改worker侧的KV cache的读写过程。对于写操作，仍可按照原来DP域所匹配的请求来存储KV cache。如下图所示，假设seq 0分配到了DP0、seq1分配到了DP1，对应存储KV值的时候，依然按照请求的原始分配关系进行保存。

在读取prefix cache的时候多一个操作步骤，即对读取完的cache进行allgather操作，保证每个DP拿到全量的序列，这样stage2依然正常运行。

这种方式适用于PD分离场景，实现了算力均衡，但KV cache有存储不均衡的问题。

挑战2：考虑如何兼容PD混部场景？

因为每个请求的KV cache在所有DP上都要存储一个副本（或者计算前对cache进行allgather）才能保证Decode阶段计算正确，要考虑解决消除冗余存储的问题。退一步，若用冗余存储的方式，则涉及scheduler中blocks预分配逻辑的修改，保证创建的blocks数量满足要求。

为了规避上述问题，我们提出一种折中方案：在stage2的过程中将请求还原。此方案既避免了冗余存储，也无需修改 KV Cache 的管理逻辑，但 stage2 的负载不均衡问题仍未解决。至于该问题的影响程度，可参考第1节中关于“无 head 维度”与“有 head 维度”计算占比的分析。

挑战3：特性带来了显存用量上升的问题如何处理？

allgather导致每个DP都要输入全量序列数据，使得输入的激活值变大，所需显存相比之前就会更大。

解决方式1：可以用CP序列并行来解决序列太长引发的激活值变大，但代码修改量大。

解决方式2：让o_proj做TP并行，采用W矩阵行切。若输入的Tensor为[m, n]，权重W为 [n, k]，输出Tensor 为[m, k]，则单个rank上面元素总数E满足：

其中S表示总序列长度，假设有n个序列要处理，总长度S = seq_0 + ... +seq_n。tp和dp分别是TP和DP策略的大小，world size是总卡数。合并公式得到：

元素总数E跟tp 和1/tp都有关，并不是tp开得越大，E就越小。所以显存占用跟TP之间是抛物线关系，挑两组参数来分析如下。

当曲线过了最小值后，tp越大，o_proj计算所需的显存越大。值得注意的是实际应用中，总序列长度S是变化的，对应的TP最佳值也会变化。

解决方式3：降低线性层的权重冗余，参考ZeRO的分片逻辑。降低down_proj、o_proj的权重冗余。以o_proj的权重存储计算为例，其切分大小如下：

采用deepseekV3 671B FP8格式，其中k是分片的数量。

3 实现与测试
我们[3]当前的实现方案是基于vLLM框架，为了用较小的改动实现收益，MLA的stage1和stage3采用SP并行，而stage2与之前保持一致，即上面提到的“stage2还原方案”。clrs97[4]同学细化了方案的具体步骤，在Ascend芯片上进行了实践测试，除了解决通用问题，还涉及与芯片相关的特殊问题处理。

3.1 跨DP混合序列并行
方案的主要执行步骤：

将所有请求数据拼接后，均匀切分并送入 stage1 进行计算；
将down_proj的输出执行allgather，随后从中选取原 DP 域所需的数据（区别点1）；
stage2在DP域内进行TP切分并行计算；
attention计算完成后，执行alltoall完成数据分发；
stage3的输入为切分后的序列，o_proj计算采用了权重分片方式（区别点2）。

代码实现

在前向计算（forward）中，首先将所有请求的数据进行拼接，并记录其原始的维度信息。

原始信息和切分信息通过context记录，包括请求、设备、计算序列之间的映射关系。这个数据也是后续请求还原的基础。

MLA计算过程中，stage1序列选择、stage2前序列的还原操作实现如下：

这种方式的不足：stage2没有实现DP间算力均衡，且需要引入额外的allgather操作；优势：修改相对简单。

数据NZ格式与ND格式转换问题：当前，传输 allgather 算子需要使用 ND[5]格式，而权重采用的NZ格式。因此在数据传输过程中需要进行格式转换以确保正确性，但该转换操作所带来的耗时不容忽视。

当前的处理方式：采用broadcast操作来完成数据分发，避免格式错误。这也是在o_proj的权重分片过程中采用broadcast一个重要原因。

其它细节参考对应的PR：https://github.com/vllm-project/vllm-ascend/pull/2493

3.2 O_proj操作抽取通用类
在面对聚合序列导致显存占用上升的问题时，采用了o_proj分片来降低影响。我们实践中选用的让o_proj的TP=1，即激活值不切分、仅对权重按层分片。

该方法符合 ZeRO 中减少线性层冗余权重的设计思想。因此，我们将这一分片操作进一步封装，提取出公共类LayerShardLinear，也为未来实现通用线性层的分片操作奠定基础。

具体步骤结合示意图：假设有n个设备、模型有3n层，则每个设备中存储3个线性层的分片参数，同时用一个buffer来缓存需要计算的数据，参数通过broadcast分发到每个rank中。为了实现计算与通信的掩盖，将buffer设置为大于1的值，实现参数的预取，比如设置buffer=x，当前计算k层，缓存获取了k+x层。

承载功能的LayerShardLinear继承自LinearBase，在基本线性层基础上叠加分片功能。

class LayerShardLinear(LinearBase):
def **init**(

# ...

# ...

        return_bias: bool = True,
        series_name: str,
        group: GroupCoordinator,
        start_layer: int,
        end_layer: int,
        layer_idx: int,
        prefetch_step: int = 0,

关键参数：start_layer、end_layer用于定义分片的起止范围，layer_idx表示当前层的索引，prefetch_step表示提前预取多少层，满足buffer size = prefetch_step + 1。承载分片数据和buffer数据的结构类分别是：LayerMetadata、SharedWindowMetadata。

@dataclass
class LayerMetadata:
"""Metadata for a layer.
"""
layer: Optional[LinearBase] # The layer object.
post_method: Callable[[
torch.nn.Module
], None] # The `process_weights_after_loading` method from the quant method.
weight: torch.Tensor # The weight tensor.
window_idx: int # The index of the window.

@dataclass
class SharedWindowMetadata:
"""Metadata for a shared window.
"""
weight: torch.Tensor # The weight tensor to be shared by layers.
data_layer_idx: int # The index of the layer this window's weight is equal to.
work: Optional[torch.distributed.Work] # The asynchronous broadcast work.
详情参考PR：https://github.com/vllm-project/vllm-ascend/pull/2931

线性层分片的另一种实现方案是对单个 linear 层的参数进行切分，每个 rank 仅保存其中一部分。在需要计算时，通过 allgather 操作将参数还原。若在 NPU 上实践该方案，则需重点考虑 allgather 操作中涉及的 NZ 格式数据传输问题。

3.3 性能收益测试与分析
我们的应用场景中序列总长度限制为4K，采用DP4、TP4的启动脚本/参数如下：

export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1

python -m vllm.entrypoints.openai.api_server --model=DeepSeek-R1-W8A8-VLLM \
 --served-model-name auto \
 --load-format=auto \
 --trust-remote-code \
 --enforce_eager \
 --distributed-executor-backend=mp \
 -tp=4 \
 -dp=4 \
 --quantization ascend \
 --additional-config '{"chunked_prefill_for_mla": true, "enable_mla_sp": true, "o_shard_parallel_size": 8, "o_shard_full_layers": 5}' \
 --port 8006 \
 --max-num-seqs 24 \
 --max-model-len 8192 \
 --max-num-batched-tokens 8192 \
 --block-size 128 \
 --gpu-memory-utilization 0.96 \
 --no-enable-prefix-caching
数据如下：

吞吐单位qps
可以看到，定长输入条件下性能有20%+收益；在随机长度的情况下有50%收益。

进一步测试DP1、TP16场景，参照基线是无SP纯开TP。该场景下无跨DP域的需求，为了测试我们的特性对性能的影响。参数调整：

--max-model-len 32768 \
--max-num-batched-tokens 32768 \
数据如下：

吞吐单位qps
可以看到在DP=1的情况下，单请求收益为负值，多请求并发下性能基本持平。

3.4 进一步工作
i. 解决 stage2 算力不均衡问题

当前方案仍存在 stage2 算力不均衡的问题，根据前文分析，该问题在长序列场景下影响更为显著。下一步：参考前文所述的 KV cache 选择性存储方案，保持各 DP 域内 KV cache 存储的一致性。仅在 prefill 阶段启用 SPTP 混合并行，而在 decode 阶段采用其他并行策略，从而使 prefill 的 stage2 实现计算均衡。

ii. 与其它特性的融合

考虑与 CP（Context Parallelism）并行等特性进行结合。CP 并行不仅能处理超长序列，若与跨 DP 的 SP 结合，还可在 attention 计算阶段避免序列维度的还原操作，将 attention 的计算方式从 TP 转换为 CP。

iii. 探索o_proj开启TP

从前面的分析可知，o_proj开TP对显存影响是非线性关系，需根据场景调整TP的size才能获得相应收益。另一种思路是在固定TP size的情况下，分析其适用的最佳序列长度范围
