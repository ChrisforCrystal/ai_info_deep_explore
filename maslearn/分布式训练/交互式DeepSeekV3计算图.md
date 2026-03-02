在大模型(LLM)的应用中会用架构图来示意网络结构（如下[1]），读者能从架构图中了解到模型的基本信息，但这种人工绘制的架构图一般不会包含计算细节，如Tensor的尺寸和操作。为了便于分析数据从输入到输出的变化细节，本文以DeepSeekV3为例介绍一种PyTorch计算图（网络）可视化的方法。

LLM架构图
使用torchvista[2]工具将PyTorch定义的DeepSeekV3模型可视化后，在notebook界面中获得能互动的计算流图，如下所示。主要特点：

每个tensor都会标记尺寸信息；
每个模块可具体到PyTorch的基本操作；
采用类定义的模块能够折叠，深度可调节；
界面上能够查看每个操作的定义；

DSV3计算图
示例地址: colab notebook demo (若无法打开，参考文末github地址)

下面分步骤介绍这个互动图的构建过程。

1 基本使用
借助的torchvista是一种交互式工具，可直接在notebook中可视化PyTorch模型（nn.Module）的前向传播。适合辅助理解模型推理的计算过程。

# 工具的安装：

!pip install --upgrade torchvista
构建一个线性层来测试工具：

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvista import trace_model

# 模型定义：

class LinearModel(nn.Module):
def **init**(self):
super().**init**()
self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# 创建模型、定义输入

model = LinearModel()
example_input = torch.randn(2, 10)

# 调用torchvista来可视化模型

trace_model(
model, # 模型
example_input, # 输入
collapse_modules_after_depth=3,
show_non_gradient_nodes=False,
forced_module_tracing_depth=None,
height=500,
export_format=None
)
输出结果：

trace_model的关键参数介绍：

collapse_modules_after_depth，模块自动折叠的深度，默认为0；
show_non_gradient_nodes ，是否将不参与梯度运算的节点显示出来；
forced_module_tracing_depth ，控制trace的层数，层数设置越少显示运算的成本越低，默认None；
height ， notebook上面图形显示框的高度；
export_format ，支持导出'png'、‘svg’、‘html’格式。

Tips：

当交互图的构建速度比较慢时，降低forced_module_tracing_depth参数的大小；
遇到非nn.Module模块显示错误时，将其替换成对应pytorch操作。
2 DeepSeekV3的可视化
用PyTorch构建DSV3时需要用到量化、FP8计算，可参考官方文档[3]搭建模型。

模型依赖一些GPU kernels，所以先在notebook中安装上triton库：

# 安装triton

!pip install --upgrade triton
然后，搭建对应的模型（代码参考：https://huggingface.co/deepseek-ai/DeepSeek-V3）。

接着，尝试先将一个block(包含MLA + FFN层)的计算过程可视化出来。

args = ModelArgs()
model = Block(0, args)
torch.set*default_dtype(torch.bfloat16)
tokens = torch.randn((1, 1024, 2048) , dtype=torch.bfloat16)
freqs_cis = precompute_freqs_cis(args)
seqlen = tokens.size(1)
freqs_cis = freqs_cis[0:seqlen]
mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu*(1)
trace_model(model, (tokens,0, freqs_cis, mask), export_format='png')
block中的子模块MLA的计算流图：

整体显示

设置模型的总层数改为2：

n_layers: int = 2 # 总层数 -> 2
n_dense_layers: int = 1
运行如下代码：

args = ModelArgs()
model = Transformer(args)
torch.set_default_dtype(torch.bfloat16)
x = torch.randint(0, args.vocab_size, (2, 128)) # batch_size: 2 sequence_length:128
trace_model(model, (x,))
获得如下所示的计算流图，能看到模型中两个block的主要差异FFN层有区别，一个是MLP、另一个是MoE。

点开MoE模块，能够看到里面计算流细节：

MoE模块
小结：交互式计算流图能够较好地帮助开发者梳理模型计算过程，提供Tensor Size的追踪、PyTorch的操作查看；本文探索了DeepSeek模型可视化，通过计算流图可直观的了解模型的一些计算细节。其它vista类似的可视化工具：tensorwatch[4]、netron[5]。

本文用例地址：

github：pytorch_vista_deepseekV3.ipynb

colab notebook：pytorch_vista_deepseekV3.ipynb
