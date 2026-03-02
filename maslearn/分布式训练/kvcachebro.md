作者：kaiyuan
链接：https://www.zhihu.com/question/1974064489159730057/answer/1974067690864928547
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

在KV cache存储方面，线性注意力（Linear Attention）相比标准注意力（Standard Attention）的主要区别是其不随着序列的增长而增长，维持一个固定值。线性注意力的KV cache有多大优势？不仅需要考虑模型的参数的差异，还要考虑一些关键推理特性（如Prefix cache、重计算等）。以标准注意力为参考系，cache的比较就是固定值与线性值大小的比较，分几步来简要讨论该问题：1 了解cache的存储结构在存储结构上面，标注注意力存储的值是Key和Value（或者KV nope和rope）；线性注意力存储的是递推式的状态值(用S(SSM)表示)、短卷积的输入数据(用Conv表示)。像SSM、GDN、KDA等模块都是该类结构。当然，有些线性注意力中没有Conv部分，仅包含S数据。在推理框架中为了便于算子读取，目前不同注意力采用了相同的cache布局方式、统一的页大小（page size)。但块（block）的大小依然取决于模型层的具体参数。<img src="https://picx.zhimg.com/50/v2-71f95d62e388c147a028ee9db6a511e6_720w.jpg?source=2c26e567" data-size="normal" data-rawwidth="999" data-rawheight="437" data-original-token="v2-7d97936b7c6c726202f48824859736ee" class="origin_image zh-lightbox-thumb" width="999" data-original="https://picx.zhimg.com/v2-71f95d62e388c147a028ee9db6a511e6_r.jpg?source=2c26e567"/>vLLM中的格式2 计算的差异对比2.1 SSM的cache计算单请求单层的S&Conv的存储大小与模型的配置参数相关，FP16/BF16数据，计算公式：S:= 2 _ num_v_heads / TP _ head_k_dim _ head_v_dimConv:= 2 _ (head_k_dim _ num_k_heads _ 2 + head_v_dim _ num_v_heads)/TP _ (conv_kernel_size - 1)其中conv_kernel_size 表示卷积核的大小。另，S&Conv计算没有考虑投机（speculative）推理，若开启投机推理，还要加上额外的存储长度。2.2 MLA/GQA的cache计算在MLA/GQA中 kv cache的存储大小与序列长度相关，计算公式：MLA:= 2 _ seq_len _ (kv_lora_rank + qk_rope_head_dim)GQA:= 2 _ 2 _ seq_len _ num_key_value_heads _ head_dim / TP通过上述的计算方式，结合模型的参数。可以计算出分界点序列长度，举两个例子（假设TP=1）：例1：Qwen3-Next-80B-A3B-Instruct，模型相关参数：# GDN:
"linear_num_key_heads": 16,
"linear_num_value_heads": 32,
"linear_value_head_dim": 128,
"linear_key_head_dim": 128,
"linear_conv_kernel_dim": 4,

# GQA:

"num_key_value_heads": 2,
"head_dim": 256,通过逐步增加序列长度，得到曲线如下，交点处的序列长度（tokens）为536。<img src="https://pica.zhimg.com/50/v2-f5339bdd71ffdd5b58c76d8f1a8cbddc_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="748" data-rawheight="404" data-original-token="v2-4186ca445031801e4a775d772c117c07" class="origin_image zh-lightbox-thumb" width="748" data-original="https://pic1.zhimg.com/v2-f5339bdd71ffdd5b58c76d8f1a8cbddc_r.jpg?source=2c26e567"/>例2：Kimi-Linear-48B-A3B-Instruct，模型相关参数：# KDA:
"head_dim": 128,
"num_heads": 32,
"short_conv_kernel_size": 4

# MLA:

     "qk_rope_head_dim": 64,
     "kv_lora_rank": 512通过逐步增加序列长度，得到曲线如下，交点处的序列长度（tokens）为975。<img src="https://picx.zhimg.com/50/v2-45de5e9ab72f55296c28581e738dddc2_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="1037" data-rawheight="428" data-original-token="v2-b00754e3c2620cd0420db75d8a663473" class="origin_image zh-lightbox-thumb" width="1037" data-original="https://picx.zhimg.com/v2-45de5e9ab72f55296c28581e738dddc2_r.jpg?source=2c26e567"/>结合两个示例的曲线可知：当序列长度超过一个交汇位置tokens数量时，线性注意力层的cache更小；不同模型的参数的交汇位置的序列长度不一样。3 Prefix cache特性由于线性注意力层保留的是最后的状态，所以一般而言前缀缓存（prefix cache）只能匹配最长序列。如果要匹配中间的结果，得让线性注意力层保存对应的S&Conv的中间状态数据。考虑的实现方案：离散保存：按照block为单位进行状态保存，例如128个tokens保存一次S&Conv。这种方式下存储cache呈现分段增长。以算换存：存储GDD/KDA的输入值hidden states，或者映射后的输入值（k/v/alpha/beta），通过重算的方式恢复S；3.1 离散保存的对比用Qwen3-Next-80B-A3B-Instruct的参数为例进行对比计算，GQA与S&Conv的对比曲线如下：<img src="https://pic1.zhimg.com/50/v2-52ebe7d6e440d8ba43e1fe25cc3d20fa_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="1078" data-rawheight="429" data-original-token="v2-cd685c917bf4a21f62ed8520d10d4e86" class="origin_image zh-lightbox-thumb" width="1078" data-original="https://pic1.zhimg.com/v2-52ebe7d6e440d8ba43e1fe25cc3d20fa_r.jpg?source=2c26e567"/>根据对比可知，在该模型下：就单层而言，S&Conv的cache增长速度大于GQA的线性增长。实际应用中，离散保存S&Conv方式必须要定期的进行cache的卸载（CPU offload）。3.2 以算换存的对比用Qwen3-Next-80B-A3B-Instruct的参数，对比三种情况：保存hidden states 保存k/v/alpha/beta参考系：离散存储S&Conv（128tokens保存一次）1和2对应的存储大小：保存hidden states := 2 * seq_len * hidden_size保存k/v/alpha/beta:= 2 * seq_len * (head_k_dim * num_k_heads + head_v_dim * num_v_heads + num_v_heads)随着序列增长，cache的变化曲线如下：<img src="https://pica.zhimg.com/50/v2-134620e2da84fc75a47a66125f4afc98_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="711" data-rawheight="391" data-original-token="v2-8bdc5f2336c2ca684cc3be0f8a27f7e6" class="origin_image zh-lightbox-thumb" width="711" data-original="https://picx.zhimg.com/v2-134620e2da84fc75a47a66125f4afc98_r.jpg?source=2c26e567"/>通过对比可知，在该模型下：保存k/v/alpha/beta的方式虽然能避免投影线性层的重复计算，但所需空间最大；存储输入hidden states的方式存储量最少，重复计算量最高；离散存储S&Conv属于存储量始终，无需重复计算。小结：KV cache的大小主要影响访存和cache传输。混合模型中当序列超过某个值后，线性注意力的cache更小，通过调节模型参数可以更好的发挥线性注意力的优势；在前缀匹配特性上，线性注意力存储cache的方案有多种，需要在计算与存储之间做平衡。若采用离散存储，需要开启CPU Offload功能，避免出现OOM。
