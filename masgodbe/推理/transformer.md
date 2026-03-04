Transformer模型已经过多次迭代后，版本越来越多、结构也越来越复杂，阅读起来不太友好，而理解transformer结构是跨入大模型的基础。本人用PyTorch实现一个简易版Transformer模型，代码就一个.py文件，可帮助初学者更好学习。Code：

Transformer One Page
github.com/CalvinXKY/transformer_one_page
本文主要讲解“One-Page”代码编写过程并对一些关键知识点进行解释。

1 整体结构
Transformer主要是由Encoder、Decoder块组成如下所示，Encoder/Decoder层的子模块基本相同可以复用。代码中参数命名尽量参考Attention Is All You Need原文。

transformer架构图
拆解下来大概有下图所示的一些主要模块，其中最关键的模块是MHA（Multi-Head Attention），以及由特色的位置编码PE(Positional Encoding)模块。

对于编码可以参照如下步骤：

步骤1：实现子模块：

Embedding
Positional Encoding
Multi Head Attention (Key)
Position wise Feed Forward
Encoder Layer
Decoder Layer
Generator
步骤2：搭建N次重复的层

Encoder
Decoder
步骤3： 组装成transformer

动图封面
编码流程
2 子模块编写
编码过程中用到的缩写介绍：

名称 ABBR 解释
bs batch size 缩写，批大小
seq_len/L 最大的 src/trg的token sequence的长度
d_model 模型尺寸参数。别称token_size/ embed_size / hidden_size
heads 头数
dk 表示K/Q的维度，一般dk=dq=dv，dk=d_model / heads
pe positional encoding 缩写
dff FFN层的内层尺寸
p_drop dropout的概率
ffn position-wise feed-forward networks 缩写
MHA multi-head attention缩写
2.1 Self attention模块
self attention模块主要是实现如下公式的计算：

Scaled Dot-Product Attention
其中Q（query）、K（key）、V（value）为输入注意力参数，dk表示K/Q的维度。按照计算结构图直接搭建attention的代码。

Scaled Dot-Product Attention计算结构图
class Attention(nn.Module):
def **init**(self, input_dim, output_dim):
super().**init**()
self.query = nn.Linear(input_dim, output_dim)
self.key = nn.Linear(input_dim, output_dim)
self.value = nn.Linear(input_dim, output_dim)
self.dk = output_dim

    # Scaled dot-product attention:
    def self_attention(self, query, key, value, mask):
        # query/key/value:  (bs,  seq_len, dk)/(bs, heads, seq_len, dk)
        # mask shape = (bs, 1, seq_len)/(bs, 1, 1, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dk)) # (bs, seq_len, seq_len)/(bs, heads, seq_len, seq_len)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # Softmax dim=-1 stands for apply the softmax along the last dimension
        attention_weights = nn.Softmax(dim=-1)(scores)  # (bs, heads, seq_len, seq_len)/(bs, seq_len, seq_len)
        attention_qkv = torch.matmul(attention_weights, value)   # (bs, seq_len, dk)/(bs, heads, seq_len, dk)
        return attention_qkv

    def forward(self, query, key, value, mask):
        # qkv shape: (bs, seq_len, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        attention_qkv = self.self_attention(query, key, value, mask)  # shape:  (bs, seq_len, d_model)
        return attention_qkv

知识点：公式计算时除以
的原因？

因为qk相乘的值（注意力得分）随着dk维度的增加而增加，当权重过大时模型会变得不稳定（梯度传递时，容易出现梯度消失或梯度爆炸现象），所以，除以根号dk使得值在合理范围内。
采用除以
相比dk优势：降低dk大小变化的影响。
2.2 Multi-head self-attention模块
Multi-head self-attention模块是在self attention的增加一个多头处理，直观理解就是相同QKV产生了多个self attion计算。代码实现上需要将QKV进行计算的维度映射。

qkv格式： (bs, seq_len, dk\*heads)

默认参数取值：d_model = 512; heads = 8; dk=dv=512/8=64

代码实现上面MultiHeadedAttention继承Attention，重写forward函数：

class MultiHeadedAttention(Attention):
def **init**(self, d_model, heads):
super().**init**(d_model, d_model)
assert d_model % heads == 0
self.dk = d_model // heads # head dimension
self.heads = heads
self.out_linear = nn.Linear(d_model, d_model)
self.sqrt_dk = torch.sqrt(torch.tensor(self.dk))

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        # qkv shape: (bs, seq_len, dk*heads)
        # dk * heads = d_model
        query = self.query(query).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
        attention_qkv = self.self_attention(query, key, value, mask)  # shape:  (bs, heads, seq_len, dk)
        #  (bs, heads, seq_len, dk) -> (bs, seq_len, dk*heads)
        reshaped = attention_qkv.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dk)
        representations_batch = self.out_linear(reshaped)
        return representations_batch

其中从masked操作是可选的，代码中直接将对应scores值置为负无穷。

mask的形状：[bs, 1, 1, seq_len] 或者 [bs, 1, seq_len, seq_len]

也可以将Attention写到MHA里面：

class MultiHeadedAttentionV2(nn.Module):
""" Write self_attention into MHA """
def **init**(self, d_model, heads):
super().**init**()
assert d_model % heads == 0
self.dk = d_model // heads # head dimension
self.heads = heads
self.qkv_nets = (nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model))
self.out_linear = nn.Linear(d_model, d_model)
self.sqrt_dk = torch.sqrt(torch.tensor(self.dk))

    # Scaled dot-product attention:
    def attention(self, query, key, value, mask):
        # query/key/value shape (bs, heads, seq_len, dk)
        # mask shape = (bs, 1, 1, seq_len) or (bs, 1, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dk  # shape: (bs, heads, seq_len, seq_len)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
        # Softmax dim=-1 stands for apply the softmax along the last dimension
        attention_weights = nn.Softmax(dim=-1)(scores)  # shape: (bs, heads, seq_len, seq_len)
        attention_qkv = torch.matmul(attention_weights, value)   # shape:  (bs, heads, seq_len, dk)
        return attention_qkv

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        # qkv shape: (bs, seq_len, dk*heads)
        # dk * heads = d_model
        query, key, value = [net(x).view(batch_size, -1, self.heads, self.dk).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]
        attention_qkv = self.attention(query, key, value, mask)  # shape:  (bs, heads, seq_len, dk)
        #  (bs, heads, seq_len, dk) -> (bs, seq_len, dk*heads)
        reshaped = attention_qkv.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dk)
        representations_batch = self.out_linear(reshaped)
        return representations_batch

知识点：多头的作用？

多头注意力机制通过并行地学习序列的不同部分，增强了模型的表示能力和泛化能力，使得模型能够更有效地处理复杂的序列数据。
2.3 Position encoding模块

位置编码是为了解决注意力QKV计算对位置信息不感知的问题。所以在进入Encoder/Decoder层前，将位置信息添加进去。具体的计算公式如下：

class PositionalEncoding(nn.Module):
def **init**(self, d_model, p_drop=None, max_seq_length=5000):
super().**init**()
self.dropout = nn.Dropout(p=p_drop) if p_drop is not None else None
position_id = torch.arange(0, max_seq_length).unsqueeze(1) # (max_seq_length, 1)
frequencies = torch.pow(10000., -torch.arange(0, d_model, 2, dtype=torch.float) / d_model)

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        pe[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions
        self.register_buffer('pe', pe)

    def forward(self, embeddings_batch):
        # embedding_batch  shape: (bs, seq_len, d_model)
        # pe shape: (max_seq_length, d_model)
        # pe shape broad_casted -> (bs, seq_len, d_model)
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.pe.shape[-1]
        positional_encodings = embeddings_batch + self.pe[:embeddings_batch.shape[1]]
        if self.dropout is not None:
            positional_encodings = self.dropout(positional_encodings)
        return positional_encodings

    def forward(self, x):
        assert x.size(-1) == self.pe.size(-1), "size d_model of x is not equal to pe's."
        # 添加位置编码到输入张量
        x = x + self.pe[:, :x.size(1), :]
        if self.dropout is not None:
          x = self.dropout(x)
        return x

知识点：为什么要用正弦余弦位置编码？

先看一下位置编码设计时需要满足的条件：

位置和编码之间能够进行唯一的映射，一个位置对应一个唯一编码；
编码值与输入的值的长度无关，即计算公式里面不要有length；
编码值必须是有界的。（提升泛化能力，测试数据的length可能超过训练数据的length）
满足这个设计的一个函数：

（
）

pos表示token的位置，公式中
是个常数，但存在如下问题：

太小，可能出现不同位置但编码值相同；
太大，词之间的距离值PE差距会比较小，位置特征不明显；
需要对固定值
进行改进，容易想到的一个实现方式：

（
）

其中i是位置编码向量的第i个值，
（模型大小）表示位置编码总长度。

但是该公式无法实现线性变化计算，如PE（pos1+pos2, 2i）拆解成PE(pos1， 2i)和PE(pos2, 2i)线性计算，因为

sin(a+b)拆解计算的时候出现了余弦计算，所以公式需要进一步改进满足线性计算要求，如下所示：

补充说明：

位置编码只是对位置的一种标记，并不是按照固定的升序或者降序的编排数字；
位置信息是由网络去理解，不产生额外干扰；
2.4 Positionwise FeedForwardNet模块

FFN的引入非线性映射relu() 增加网络的非线性理解能力，提高了模型表达能力。FFN构成：两个线性层+ 激活函数。原文的解释如下：

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
公式：

输入/输出参数：[bs, seq_len, d_model]

默认情况下：d_model=512；dff (hidden_dim)=2048

代码实现：

class PositionwiseFeedForward(nn.Module):
def **init**(self, d_model, dff=2048):
super().**init**()
self.linear1 = nn.Linear(d_model, dff)
self.linear2 = nn.Linear(dff, d_model)
self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.relu(self.linear1(representations_batch)))

2.5 Embedding模块

输入参数 [bs, seq_len]

输出参数 [bs, seq_len, d_model]

class Embedding(nn.Module):
def **init**(self, vocab_size, d_model):
super().**init**()
self.embeddings_layer = nn.Embedding(vocab_size, d_model)
self.sqrt_d_model = torch.sqrt(torch.tensor(d_model))

    def forward(self, tokens):
        assert tokens.ndim == 2
        # tokens shape: (bs, seq_len)
        # embeddings shape: (bs, seq_len, d_model), every token id has associated vector
        embeddings = self.embeddings_layer(tokens)
        # Paper P-5, Chapter 3.4 "Embeddings and Softmax": multiply the embedding weights by the square root of d_model
        embeddings = embeddings * self.sqrt_d_model
        return embeddings

注意在embedding层输出后面乘以了一个
，参看原文5页3.4章.

2.6 Add&Layer Normalization模块

Add&Norm主要是进行残差链接以及进行归一化处理，其中归一化用的是Layer Normalization，LayerNorm是计算每个词向量的维度上归一操作。计算针对的是特征维度：d_model，在MHA里d_model有多头拆分，但最后会cat成一个整体所以不影响LN计算。

输入/输出参数：[bs, seq_len, d_model]

实现代码如下：

class AddNormLayer(nn.Module):
def **init**(self, d_model, p_prob):
super().**init**()
self.LN = nn.LayerNorm(d_model)
self.dropout = nn.Dropout(p=p_prob)

    def forward(self, representations_batch, sublayer_module):
        return representations_batch + self.dropout(sublayer_module(self.LN(representations_batch)))

对Layer Normalization的理解可以借助如下示例：

import torch

# 定义输入张量

x = torch.tensor([
[[0.2, 0.4, 0.6],
     [0.3, 0.5, 0.2],
     [0.1, 0.9, 0.7]],

    [[0.6, 0.2, 0.8],
     [0.4, 0.7, 0.3],
     [0.2, 0.6, 0.4]]

])

# 计算均值和标准差

mean = x.mean(dim=-1, keepdim=True) # 在最后一个维度上计算均值
std = x.std(dim=-1, keepdim=True) # 在最后一个维度上计算标准差

# 执行 Layer Normalization

epsilon = 1e-8 # 用于数值稳定性的小值
normalized_x = (x - mean) / (std + epsilon)

# 打印结果

print(normalized_x)
LN和BN的计算区别：

import numpy as np

def batch_normalization(x, epsilon=1e-8):
mean = np.mean(x, axis=0, keepdims=True) # 维度0
std = np.std(x, axis=0, keepdims=True)
normalized_x = (x - mean) / (std + epsilon)
return normalized_x

def layer_normalization(x, epsilon=1e-8):
mean = np.mean(x, axis=-1, keepdims=True) # 最后一个维度
std = np.std(x, axis=-1, keepdims=True)
normalized_x = (x - mean) / (std + epsilon)
return normalized_x
2.7 Generator模块

在decoder最后输出的时候经历了一个线性层和softmax处理，这两个操作最后会生产概率结果，这里把这个模块叫做Generator。它最后用的是一个LogSoftmax操作。

输出参数：[bs, seq_len, d_model]

输出参数：[bs, seq_len, vocab_size]

代码实现：

class Generator(nn.Module):
def **init**(self, d_model, vocab_size):
super().**init**()
self.linear = nn.Linear(d_model, vocab_size)
self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, trg_representations_batch):
        # trg_representations_batch shape: (bs, seq_len, d_model)
        # output shape: (bs, seq_len, vocab_size)
        return self.log_softmax(self.linear(trg_representations_batch))

2.8 Encoder&Decoder layer

Encoder Layer和Decoder Layer实现类似，需要注意的是对于Add&Norm 这种重复的模块，可以用replicate_module函数简化定义抒写，用lambda操作来简化调用抒写；

如DecoderLayer代码，decoder_src_attention 模块可以用lambda函数保持srb内容。Encoder的参看：transformer.py L127-L146

class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, p_prob):
        super().__init__()
        self.sublayers = replicate_module(AddNormLayer(d_model, p_prob), 3)
        self.trg_multi_headed_attention = MultiHeadedAttention(d_model, heads)
        self.src_multi_headed_attention = MultiHeadedAttention(d_model, heads)
        self.ffn = PositionwiseFeedForward(d_model)
        self.d_model = d_model

    def forward(self, trg_representations_batch, src_representations_batch, trg_mask, src_mask):
        srb = src_representations_batch
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)
        decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        # Self-attention MHA sublayer followed by a source-attending MHA and point-wise feed forward net sublayer
        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)
        trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)
        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.ffn)

        return trg_representations_batch

3 Transformer实现
3.1 Encoder&Decoder模块

Encoder和Decoder模块主要是重复Encoder&Decoder layer 多次（N次），Decoder代码：

class Decoder(nn.Module):
def **init**(self, decoder_layer, number_of_layers):
super().**init**()
self.decoder_layers = replicate_module(decoder_layer, number_of_layers)
self.LN = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, trg_embeddings_batch, src_representations_batch, trg_mask, src_mask):
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch,
                                                      trg_mask, src_mask)
        return self.LN(trg_representations_batch)  # Using LN. not mentioned explicitly in the paper.

Decoder和Encoder模块在所有layer计算完成后，最后进行了一次LayerNorm操作，这个在原文中没有提到，这样能够增加模型稳定性。当然，Decoder接的Generator里面有LN，所以此处不一定必要。

Encoder代码参看，transformer.py L186-L196

3.2 模块拼装
Transformer的整体就是上面的模块组合起来。代码角度一共需要创建7个大模块，如下图所示：

最后在forward中，这些模块串接起来：

    def forward(self, src_token_ids_batch, trg_token_ids_batch, src_mask, trg_mask):
        src_representations_batch = self.encode(src_token_ids_batch, src_mask)
        trg_log_probs = self.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)
        return trg_log_probs

    def encode(self, src_token_ids_batch, src_mask):
        src_embeddings_batch = self.src_embedding(src_token_ids_batch)  # (bs, seq_len) -> (bs, seq_len, d_model)
        src_embeddings_batch = self.src_pos_embedding(src_embeddings_batch)
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        return src_representations_batch

    def decode(self, trg_token_ids_batch, src_representations_batch, trg_mask, src_mask):
        trg_embeddings_batch = self.trg_embedding(trg_token_ids_batch)  # (bs, seq_len) -> (bs, seq_len, d_model)
        trg_embeddings_batch = self.trg_pos_embedding(trg_embeddings_batch)
        trg_representations_batch = self.decoder(trg_embeddings_batch, src_representations_batch, trg_mask, src_mask)

        # linear projection followed by log softmax
        trg_log_probs = self.generator(trg_representations_batch) # (bs, seq_len, d_model) -> (bs, seq_len, vocab_size)

        # (bs*seq_len, vocab_size) format for passing it into KL div loss
        trg_log_probs = trg_log_probs.reshape(-1, trg_log_probs.shape[-1]) # (bs, seq_len, vocab_size) -> (bs*seq_len, vocab_size)

        return trg_log_probs

4 OnePage运行
为了验证transformer代码是否编写正确可以用一些数据shape来进行判断，这里编写了三个测试用例在代码结尾transformer.py，用于测试PE计算、MHA计算、transformer模型。transformer.py 在python环境（需要安装torch包）下运行：

python transformer.py
输出内容如下所示：

动图封面
论文位置：Attention Is All You Need

文章代码位置：transformer_one_page/blob/main/transformer.py

补充问题：

GPT类模型（只有decoder模块）的输入输出是什么形状的？含义如何理解？

输入输出的seq保持一致，n_seq <= max_seq，(不考虑batch_size):

输入：[n_seq, n_embd]
输出logits：[n_seq, n_vocab]
假设输入序列为[token1, token2, token3]，输出logits形状为[3, n_vocab]。其中：

output[0]表示在token1位置，下一个token的概率分布。
output[1]表示在token2位置，下一个token的概率分布。
output[2]表示在token3位置，下一个token的概率分布。
GPT类模型loss计算如何计算？

假设输入序列为["你", "好"]，模型的输出是一个形状为[2, n_vocab]的概率分布矩阵。目标序列是["好", "<eos>"]。对于第一个位置，我们使用输出p1和目标"好"来计算loss；对于第二个位置，我们使用输出p2和目标"<eos>"来计算loss。

预测的时候带kv-cache，输入和输出形状？

长度为1（不考虑batch_size）：

输入：[1, n_embd]
输出logits：[1, n_vocab]
