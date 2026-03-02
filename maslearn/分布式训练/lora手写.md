Multi-LoRA(Low-Rank Adaptation)作为大模型部署的常见方案，解决了多场景下训/推成本高的问题。本文结合一个分类案例讲解LoRA原理，读者参考本文提供的notebook用例[1](支持在线运行[2]），按照步骤实践后基本能掌握LoRA的概念与Multi-LoRA的构建过程。

目录:

1 基本原理
1.1 问题与方案
1.2 低秩分解的原理
1.3 低秩分解的代码实践
2 LoRA
2.1 计算公式
2.2 LoRA训练/推理实践
3 Multi-LoRA
3.1 LoRA的变体
3.2 Multi-LoRA方案
3.3 Multi-LoRA训/推实践
4 Multi-LoRA的性能优化
关键问题：

什么是低秩分解，对计算有什么影响？
LoRA特点是什么，一般应用在哪些层上？
Multi-LoRA原理是什么，如何提升其性能？
1 基本原理
1.1 问题与方案
为了让大模型在细分领域要取得更好的效果，会用领域数据进行微调训练，且微调模型时期望用少量的训练步骤完成对权重的更新。微调一个大模型要有匹配的硬件资源和足够的训练时间。对于动辄百亿参数的大模型而言，可能出现如下问题：

硬件资源无法支持基础模型的训练。如显存不足、算力太低（训练时间过长）；
训练不收敛或者效果不佳；
大模型的通用能力下降。
既然大模型的全量调参成本高，是否能仅微调部分参数达到与全量微调的效果？这个问题已有不少的研究，如：

适配器(Adapter[3])：一种在模型中插入新层的方式，仅训练插入的适配器；
前缀调优（Prefix Tuning[4]）：给Attention KV层中添加一个前缀，并只训练这个附加的前缀参数。类似的方式还有提示词调优（Prompt Tuning[5]）；

局部训练：仅训练Transformer的LayerNorm参数[6]，或者仅训练Bias(BitFit[7])；
低秩适配（LoRA）：给模型增加降秩权重，且仅训练该新增的权重；
当然这些方法也可以混合使用[8]。

参数高效迁移学习 (PETL，parameter-efficient transfer learning)
上述迁移学习的方式各有特点，此处不展开过多讨论，主要聚焦LoRA方法的相关内容。

1.2 低秩分解的原理
LoRA原理涉及的关键知识：任意矩阵都能进行奇异值分解（Singular Value Decomposition，SVD）；当矩阵是不满秩矩阵（Rank-deficient Matrix）时，可以用低秩的分解矩阵来代替原矩阵。

具体展开说明。对于一般矩阵
通过SVD计算能够得到三个子矩阵，公式如下：

其中
，
，
，矩阵的秩数满足：

是一个对角矩阵，对角上非零元素个数等于秩数，当W为不满秩矩阵时，
对角上面会出现零元素。既然有零元素，可去除分解矩阵的零元素，若对角阵为对称矩阵则简化后的矩阵尺寸变为：

，
，

这里举个3x2矩阵分解的例子，如下图所示，将矩阵W进行SVD处理，得到分解矩阵。其中3x2的对角矩阵最后一行必为0，所以可以简化表达；进一步若还存在‘0’的对角元素，可以进一步简化。简化后的分解矩阵乘积依然等于原矩阵。

优势：当矩阵的尺寸（m,n）较大时，分解矩阵的特点是元素个数相比原矩阵的更少，r越小元素越少。比如当r=1，m=n=1000时，原矩阵元素个数为1000,000，分解矩阵元素总数为2001，比值小于0.5%。参数量少带来好处是：计算量少、存储量少。

1.3 低秩分解的代码实践
这里我们通过一个简单的乘法示例来验证分解矩阵的特点。先创建一个非满秩的矩阵W并进行SVD计算，接着建立B、A矩阵，最后定义一个乘加运算，对比原矩阵与分解矩阵的计算差异。

step1：创建一个非满秩矩阵
import torch
import numpy as np

d, k = 10, 10

# 创建一个非满秩矩阵(a rank-deficient matrix)

W_rank = 2
W = torch.randn(d,W_rank) @ torch.randn(W_rank,k)

W_rank = np.linalg.matrix_rank(W)
打印相关结果：

step2：进行SVD分解，构建B、A矩阵。

# 对W进行SVD处理：(W = UxSxV^T)

U, S, V = torch.svd(W)

# 对于对角矩阵S保留前rank个数据即可，相应的U和V也只要保存前rank行的数据。

U_r = U[:, :W_rank]
S_r = torch.diag(S[:W_rank])
V_r = V[:, :W_rank].t()

# 定义： B = U_r \* S_r；A = V_r

B = U_r @ S_r
A = V_r
打印相关参数：

step3：构建一个线性层，对比计算差异：

# 创建一个线性运算的输入， y = Wx + b

bias = torch.randn(d)
x = torch.randn(d)

# 原始计算 y = Wx + bias

y = W @ x + bias

# 分解矩阵计算 y' = (B\*A)x + bias

y_prime = (B @ A) @ x + bias

print(f"The result is allclose: {torch.allclose(y, y_prime)}")
可以看到打印输出为True，该例中W的元素个数为100，B和A的元素总数为40。

低秩分解降低了元素总数，且不改变计算结果；如果分解运算为一次运算，则在算量上面也更少。

2 LoRA
2.1 计算公式
LoRA正是用低秩分解矩阵的特点来降低微调矩阵的元素个数，原矩阵为
，微调矩阵为
，输出的定义[9]：

其中
，
是
的分解矩阵表达。

微调时
冻结（不参与训练）、仅微调分解矩阵 A和B ，因为
，所以需要训练的参数相比直接训练原矩阵少很多。

2.2 LoRA训练/推理实践
选取数字0~9手写体识别的训练场景，数据采用MNIST。训练一个3层的MLP，让其具备数字手写体识别的能力。为了体现LoRA的作用，需要对数据集进行处理，先全量训练，再增加LoRA微调。大致步骤如下：

step1：构建主模型并训练，训练数据集去掉数字‘1’；
step2：测试主模型的识别能力；
step3：创建LoRA层；
step4：主模型的参数冻结，用数字‘1’的数据进行微调；
step5：测试LoRA模型，观测数据‘1’识别度差异。

构建一个简单的模型：

# 创建一个全连接的网络用于手写体识别：

class MLP(nn.Module):
def **init**(self, hidden_size_1=1000, hidden_size_2=2000):
super(MLP,self).**init**()
self.linear1 = nn.Linear(28\*28, hidden_size_1)
self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
self.linear3 = nn.Linear(hidden_size_2, 10)
self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

net = MLP().to(device)

模型结构
接着定义模型的训练函数、测试函数：

# 训练函数定义：

def train(train_loader, net, epochs=5, total_iterations_limit=None):
cross_el = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
total_iterations = 0
for epoch in range(epochs):
net.train()
loss_sum = 0
num_iterations = 0
data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
if total_iterations_limit is not None:
data_iterator.total = total_iterations_limit
for data in data_iterator:
num_iterations += 1
total_iterations += 1
x, y = data
x = x.to(device)
y = y.to(device)
optimizer.zero_grad()
output = net(x.view(-1, 28\*28))
loss = cross_el(output, y)
loss_sum += loss.item()
avg_loss = loss_sum / num_iterations
data_iterator.set_postfix(loss=avg_loss)
loss.backward()
optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

# 测试函数定义：

def test(model=net):
correct = 0
total = 0
wrong_counts = [0 for i in range(10)]
with torch.no_grad():
for data in tqdm(test_loader, desc='Testing'):
x, y = data
x = x.to(device)
y = y.to(device)
output = model(x.view(-1, 784))
for idx, i in enumerate(output):
if torch.argmax(i) == y[idx]:
correct +=1
else:
wrong_counts[y[idx]] +=1
total +=1
result_str = ""
for i in range(len(wrong_counts)):
result_str += f'The wrong counts of digit {i}: {wrong_counts[i]}\n'
print(f'\nAccuracy: {round(correct/total, 3)}\n{result_str}')
step1：构建主模型并训练，训练数据集去掉数字‘1’。

# 下载MNIST手写体数字识别的数据

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 加载手写体数据：

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) # 训练集
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) # 测试集

# 去掉数字‘1'的数据，模型对‘1'的识别率存在问题

exclude_indices = torch.tensor([False if x == 1 else True for x in mnist_trainset.targets])
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]

# 训练模型：

train(train_loader, net, epochs=1, total_iterations_limit=2000)
step2：测试主模型的识别能力。
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
test()

可以看到，数字‘1’在测试集上表现不佳。

step3：创建LoRA层；

# 定义LoRA对权重修改：

class LoRAParametrization(nn.Module):
def **init**(self, features*in, features_out, rank=1, alpha=1, device='cpu'):
super().**init**() # 低秩矩阵的定义：
self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
nn.init.normal*(self.lora_A, mean=0, std=1)

        # 参考论文：https://arxiv.org/pdf/2106.09685 4.1节 设置一个比例系数：
        self.scale = alpha / rank
        # LoRA开关：
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

LoRA层对权重的计算
将LoRA层注册到模型中：

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1): # LoRA仅修改W，忽略bias修改。
features_in, features_out = layer.weight.shape
return LoRAParametrization(
features_in, features_out, rank=rank, alpha=lora_alpha, device=device
)

# 保存一份原始权重数据，用于后续校验

original_weights = {}
for name, param in net.named_parameters():
original_weights[name] = param.clone().detach()

# 注册LoRA权重到原始层中：

parametrize.register_parametrization(
net.linear1, "weight", linear_layer_parameterization(net.linear1, device)
)
parametrize.register_parametrization(
net.linear2, "weight", linear_layer_parameterization(net.linear2, device)
)
parametrize.register_parametrization(
net.linear3, "weight", linear_layer_parameterization(net.linear3, device)
)

# 定义LoRA开关函数：

def enable_disable_lora(enabled=True):
for layer in [net.linear1, net.linear2, net.linear3]:
layer.parametrizations["weight"][0].enabled = enabled
打印原始参数和添加LoRA参数的对比，LoRA占比仅0.242%。

step4：用数字‘1’数据微调；

# 将原始权重冻结：

for name, param in net.named_parameters():
if 'lora' not in name:
print(f'Freezing non-LoRA parameter {name}')
param.requires_grad = False

# 过滤数据，仅保留‘1'的数据：

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = torch.tensor([True if x == 1 else False for x in mnist_trainset.targets])
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# 用数据‘1'训练带有LoRA的模型：

train(train_loader, net, epochs=1, total_iterations_limit=100)
step5：测试LoRA模型，观测数据‘1’识别度差异。

# 测试有LoRA的情况:

enable_disable_lora(enabled=True)
test()
打印正确率，找到数字‘1’的错误个数，相比原模型明显降低了。

与原始模型的测试输出的正确率进行一个对比，除了数字‘1’以外其它数字的识别精度均下降。

代码地址：LoRA_to_Multi_LoRA.ipynb

LoRA特点小结

优点：

LoRA采用了横向扩展参数的方式，训练时原模型参数冻结、仅微调扩展参数，扩展参数采用低秩矩阵，保证了较低的参数量。
实践证明了LoRA的有效性，甚至能让小模型微调能达到大模型的水平[10]。
LoRA的适配方式能够保证各个垂直领域解耦训练，互不干扰。
不足：

分解矩阵B、A的秩小于原矩阵W，表达能力弱，导致LoRA的效果可能弱于全量微调；
当主模型参数量比较大且r取值不能太小时，LoRA训练成本依然很高。
3 Multi-LoRA
通过前面介绍可知，LoRA是一种单类场景的微调方案。但多业务场景下的LoRA微调模型的部署该如何进行，是每个场景都部署一个LoRA适配器（adapter）加一个基础模型吗？这涉及多场景下LoRA的混合部署。

Multi LoRA概念示意
在分析这个问题前，先了解LoRA算法一些变体，分析不同类型的LoRA适配器是否可以混合。

3.1 LoRA的变体
因为在参数量较大的模型下，LoRA依然存在训练成本高、精度不理想的问题，所以出现了对LoRA改进的研究：

qLoRA[11]：通过4位量化预训练模型、两次量化（Double Quantization）、显存CPUoffload等方法极大降低了显存空间；相比LoRA仅对Q/K/V的投影层使用adapter，qLoRA所有全连接层都用adapter，精度有提升。

方案对比
DoRA(Weight-Decomposed Low Rank Adaptation)[12]：将预训练权重分解为幅度（Magnitude）和方向（Direction）两部分，主要微调方向部分，旨在提升微调性能与稳定性。
LoRA-FA[13]：固定LoRA的A矩阵，只训练B矩阵，期望达到不损失效果的情况下减少计算量和内存开销。
AdaLoRA[14]：自适应LoRA参数调整，针对不同层、不同矩阵调整对应秩数。与之类似的还有Delta-LoRA[15]。

通过这些变体的LoRA原理可知，不是所有类型的LoRA都能混合使用，取决于权重的处理以及计算公式；但很显然同类型的LoRA是可以多个一起部署并共享基础权重的。

3.2 Multi-LoRA方案
对于推理部署而言其资源利用率是个重要指标。在实践中，如果单个模型只适配一种LoRA适配器，可能因请求量少而出现资源利用率低的情况。同时，通过LoRA原理分析可知：1. LoRA适配器的参数量一般比较小，2. 微调时不改变原模型权重，推理时仅在上面叠加一个适配器的权重值即可。所以，推理中产生了Multi-LoRA服务方案：不同场景的适配器共用一个基础模型。

Multi-LoRA计算示意
Multi-LoRA运算与LoRA一样，所有请求的数据会打包成batch送给推理服务，不同的是在推理框架中，会根据每个请求的场景需求用不同的适配器计算。

动图封面
Multi-LoRA优势：

同一个部署能够支撑不同的LoRA业务场景；
对于请求种类多、请求数量少的应用场景而言，可极大提升GPU的资源利用率；
不足：

每个LoRA适配器都需要额外显存空间；
相比单LoRA而言，因要协同多个适配器，有附加的计算成本。
3.3 Multi-LoRA训/推实践
还是用MNIST手写体识别案例实践，在LoRA基础上对步骤进行调整。

step1：构建主模型并训练，训练数据集去掉数字‘0’、‘1’、‘2’；
step2：测试主模型的识别能力；
step3：创建Multi-LoRA层；
step4：主模型的参数冻结，分别用数字‘‘0’、‘1’、‘2’数据对不同适配器进行微调；
step5：测试LoRA模型，观测数据‘0’、‘1’、‘2’识别度差异。

Multi-Lora实践场景
step1：构建主模型并训练，训练数据集去掉数字‘0’、‘1’、‘2’；

# MLP与前面LoRA相同

net_for_multi = MLP().to(device)

# 过滤数据，去掉‘0'、‘1'、‘2'的数据：

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = torch.tensor([False if x in [0, 1, 2] else True for x in mnist_trainset.targets])
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

train(train_loader, net_for_multi, epochs=1, total_iterations_limit=2000)
step2：测试主模型的识别能力；
test(model=net_for_multi)
结果打印如下所示，可以看到未训练的数字识别效果都很差

step3：创建Multi-LoRA层；

# 重新定义参数更新函数

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1): # LoRA仅修改W，忽略bias修改。
features_in, features_out = layer.weight.shape
return MultiLoRAParametrization(
features_in, features_out, rank=rank, alpha=lora_alpha, device=device
)

# 注册LoRA权重到原始层中：

parametrize.register_parametrization(
net_for_multi.linear1, "weight", linear_layer_parameterization(net_for_multi.linear1, device)
)
parametrize.register_parametrization(
net_for_multi.linear2, "weight", linear_layer_parameterization(net_for_multi.linear2, device)
)
parametrize.register_parametrization(
net_for_multi.linear3, "weight", linear_layer_parameterization(net_for_multi.linear3, device)
)

# 定义LoRA开关函数：

def select_lora(enabled=True, digit_value=0):
for layer in [net_for_multi.linear1, net_for_multi.linear2, net_for_multi.linear3]:
layer.parametrizations["weight"][0].enabled = enabled
layer.parametrizations["weight"][0].digit_value = digit_value
step4：主模型的参数冻结，分别用数字‘‘0’、‘1’、‘2’数据对不同适配器进行微调；

# 将原始权重冻结：

for name, param in net_for_multi.named_parameters():
param.requires_grad = False

# 对每个LoRA适配器进行微调：

for digit in [0 , 1, 2]:

print(f"Training the LoRA adapter: {digit}")
select*lora(enabled=True, digit_value=digit)
for name, param in net_for_multi.named_parameters():
if f'digit*{digit}' in name:
param.requires_grad = True

# 过滤数据，仅保留选定数字的数据：

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = torch.tensor([True if x == digit else False for x in mnist_trainset.targets])
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# 训练对应的LoRA适配器：

train(train_loader, net_for_multi, epochs=1, total_iterations_limit=100)
step5：测试LoRA模型，观测数据‘0’、‘1’、‘2’识别度差异。

# 打印未开LoRA的测试输出结果：

select_lora(enabled=False)
print("The original model test result:")
test(model=net_for_multi)

# 分别开启不同的LoRA适配器观测结果：

for digit in [0 , 1, 2]:
select*lora(enabled=True, digit_value=digit)
print(f"The LoRA_adapter_digit*{digit} test result:")
test(model=net_for_multi)
结果打印输出如下，可以看到不同的适配器在对应数字微调上都取得了提升效果。

4 Multi-LoRA的性能优化
Multi-LoRA计算的过程中不同LoRA适配器的AB矩阵不同，根据前面提到的公式4可知，对于数据
的LoRA计算如下：

不同请求的数据可以组batch，
，对于公式5的前半部分
的计算，由于权重相同，所以可以合并运算变为
，采用常规的GEMM运算即可；而后半部分计算由于参数各不相同，无法使用已有批处理计算。定义
，最终结果为
有：

对于公式6的求解可以用循环的方式依次计算，但这样计算会非常低效。

Multi-LoRA性能优化问题就变为了怎么样提升
的计算效率？

在GPU推理部署应用中有一种优化方案：Punica[16]，提出了一个分段聚合矩阵矢量乘法（Segmented Gather Matrix-Vector Multiplication，SGMV）的方式。

具体而言就是将
的计算拆成两步， 然后每一步都进行数据的聚合，并分两步下发。

将来自多个不同LoRA模型的请求打包成一个批次，在执行基座模型标准计算的同时，高效地从内存中各自对应的LoRA适配器矩阵（A和B），并完成增量计算。

Multi-LoRA支持批处理计算后，Punica研究中测试的性能对比[17]如下：

小结：解决Multi-LoRA的运算效率低的问题后，Multi-LoRA通过单一部署实现多业务场景的统一支撑，在请求种类繁多、单类请求数量偏少的场景中，可极大优化 GPU 资源利用率。
