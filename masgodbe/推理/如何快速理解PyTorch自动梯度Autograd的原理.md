作者：kaiyuan
链接：https://www.zhihu.com/question/1981438452038922346/answer/1988169697171100179
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

核心原理似乎是一道编程题：在无环有向图(Directed Graph)中， 每个节点记录了所有前向节点。自动梯度相当于：从叶子结点出发，使用拓扑排序遍历所有父节点，每个节点上存有一个函数，当遍历该节点时，调用函数更新节点上的一个数值。题中提到的“一个函数”是反向求导函数，“一个数值”即梯度。原理大概就是这个，下面展开讲一下。先看两个基本问题：1、什么是无环有向图的拓扑排序？举个例子，我们要学习一些课程，课程之间有些依赖关系，如下图所示。<img src="https://picx.zhimg.com/50/v2-28ef6507b35c8188e07fe8607dde1ae2_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="915" data-rawheight="471" data-original-token="v2-837f11a7620d022b8e74dc343aaec43e" class="origin_image zh-lightbox-thumb" width="915" data-original="https://picx.zhimg.com/v2-28ef6507b35c8188e07fe8607dde1ae2_r.jpg?source=2c26e567"/>假设每次只能学一门课程，我们可以这样学习：<img src="https://picx.zhimg.com/50/v2-fb84f777887a9557649c48edc52dbfae_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="1190" data-rawheight="371" data-original-token="v2-d2555af824f2abee4949b0f1c5311e6d" class="origin_image zh-lightbox-thumb" width="1190" data-original="https://pic1.zhimg.com/v2-fb84f777887a9557649c48edc52dbfae_r.jpg?source=2c26e567"/>这些例举的方式就是拓扑排序的结果。因为没有依赖关系的节点可以调整位置，所以拓扑排序的解可以不唯一。2、计算图是什么？数据结构中的计算图包括节点、边、数据，深度学习计算图的概念与之类似，但表示上略有差异：举个例子y = x + 3，其中x 、y、3 是节点/数据， “+” 是边。计算图如下：<img src="https://pica.zhimg.com/50/v2-38322b5dde082a9b114409e1ec1843a4_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="662" data-rawheight="235" data-original-token="v2-5ca5082b3a20a1bdca5501dbc3b67ad8" class="origin_image zh-lightbox-thumb" width="662" data-original="https://pic1.zhimg.com/v2-38322b5dde082a9b114409e1ec1843a4_r.jpg?source=2c26e567"/>PyTorch的自动梯度求解(Autograd)功能是该框架受欢迎的一个重要因素，开发者只需要搭建好网络，调用一个backward函数就能完成梯度的反向求解[1]，但其实现autograd功能的源代码却超过千行，对于一些刚入门的人来说，理解起来比较耗费时间。通过一个简单例子来了解autograd基本原理，内容：梯度计算的相关概念；如何利用计算图逐步求解梯度；用C++实现autograd的基本过程。示例的定义定义一个从输入x到loss的计算示例：(1)x=−2x = -2 \tag{1}x = -2 \tag{1}(2)z=2∗x z = 2 _ x \tag{2} z = 2 _ x \tag{2}(3)y1=z+5 y1 = z + 5 \tag{3} y1 = z + 5 \tag{3}(4)y2=relu(z∗z)+x+3 y2 = relu(z _ z) + x + 3 \tag{4} y2 = relu(z _ z) + x + 3 \tag{4}(5)loss=y1−y2+y2∗x loss = y1 - y2+y2 _ x \tag{5} loss = y1 - y2+y2 _ x \tag{5}可以把公式1~5画成计算图，数值用val表示，梯度用grad表示：<img src="https://picx.zhimg.com/50/v2-806330292e731bf7e352dcbfd178603e_720w.jpg?source=2c26e567" data-size="normal" data-rawwidth="861" data-rawheight="313" data-original-token="v2-806330292e731bf7e352dcbfd178603e" class="origin_image zh-lightbox-thumb" width="861" data-original="https://pic1.zhimg.com/v2-806330292e731bf7e352dcbfd178603e_r.jpg?source=2c26e567"/>计算图1 若用torch实现，包含反向计算，代码如下：import torch
x = torch.tensor([-2], dtype=torch.float32, requires*grad=True)
z = 2 * x
y1 = z + 5
y2 = (z * z).relu() + x + 3
loss = y1 - y2 + y2 \* x
loss.backward()可以将计算值和梯度都打印出来（完整代码见附件）：x: -2.0 x.grad: 64.0
z: -4.0 z.grad: 25.0
y1: 1.0 y1.grad: 1.0
y2: 17.0 y2.grad: -3.0对于正向计算的值（value）比较好计算，但梯度（grad）是如何得到？1 理论公式假设每个计算数据由两个部分组成(value, grad) ，value存储数值正向的数值，grad是反向运算得到的梯度值。 对于单步的梯度计算满足公式：(6)∇xnew=∇xold+∂(y)∂(x)∇y∇x*{new} = ∇x*{old} + \frac{\partial(y)}{\partial(x)} ∇ y \tag{6}∇x*{new} = ∇x*{old} + \frac{\partial(y)}{\partial(x)} ∇ y \tag{6}其中∇xnew∇x*{new}∇x*{new} 待求参数的梯度值， ∇xold∇x*{old}∇x*{old} 当前梯度值， ∂(y)∂(x)\frac{\partial(y)}{\partial(x)} \frac{\partial(y)}{\partial(x)} 是正向函数 y=f(x)y=f(x)y=f(x) 的求导函数， ∇y∇ y∇ y 是反向过来的梯度值。对于 f=f1+f2+⋯+fk f = f_1 + f_2 + \dots + f_k f = f_1 + f_2 + \dots + f_k 的求导公式为：(7)∂f∂xi=∂f1∂xi+∂f2∂xi+⋯+∂fk∂xi\frac{\partial f}{\partial x_i} = \frac{\partial f_1}{\partial x_i} + \frac{\partial f_2}{\partial x_i} + \dots + \frac{\partial f_k}{\partial x_i} \tag{7}\frac{\partial f}{\partial x_i} = \frac{\partial f_1}{\partial x_i} + \frac{\partial f_2}{\partial x_i} + \dots + \frac{\partial f_k}{\partial x_i} \tag{7}通过公式7可知，对于复杂的公式可拆成多个子式线性叠加。结合公式6和7，当需要计算x梯度时，先找到与x相关的所有正向计算，然后构建每个正向计算的反向公式并计算梯度分量，将梯度分量进行叠加得到最终结果。运算可以依次迭代运行，也可以并发执行。举个例子，如下图所示当计算x_0的梯度时，需要按照公式6，依次用y_0、y_1和y_2的更新x_0_grad。<img src="https://picx.zhimg.com/50/v2-3656ab08e420484acaaf167367f7d7bd_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="899" data-rawheight="424" data-original-token="v2-2ececf663990cdfeab7b69f5aebe41b8" class="origin_image zh-lightbox-thumb" width="899" data-original="https://pic1.zhimg.com/v2-3656ab08e420484acaaf167367f7d7bd_r.jpg?source=2c26e567"/>假设 y_1 = 2 \* x_0 - x_1 ， y = (1 , 3) x= (1, 2)，有 φ(y)φ(x)=2\frac{\varphi(y)}{\varphi(x)} = 2\frac{\varphi(y)}{\varphi(x)} = 2 ， ∇y=3∇ y = 3∇ y = 3 ，代入公式6得：∇x0new=2+2∗3=8∇x*{0new} = 2 + 2 _ 3 = 8∇x\_{0new} = 2 + 2 _ 3 = 8 求解中可以发现，计算x_0的梯度并不需要分析x_1与y_0/y_1/y_2之间梯度求解的关系，这样使得运算能解耦。2 用图运算（依次迭代）图运算过程先计算正向得到value，然后再逐步迭代更新grad。步骤1：正向求解 value(val)更新：x(-2, 0) ; z(-4, 0); y1(1, 0) ; y2(17, 0) ; loss(-50, 0); <img src="https://picx.zhimg.com/50/v2-400139280ff480d47b2519b6c373382e_720w.jpg?source=2c26e567" data-size="normal" data-rawwidth="861" data-rawheight="313" data-original-token="v2-400139280ff480d47b2519b6c373382e" class="origin_image zh-lightbox-thumb" width="861" data-original="https://pic1.zhimg.com/v2-400139280ff480d47b2519b6c373382e_r.jpg?source=2c26e567"/>正向求解步骤2：反向求解结合公式6更新梯度：作为触发条件loss梯度为1，即loss_grad = 1，更新loss(-50, 1); 因为有 φ(loss)φ(y1)=1\frac{\varphi(loss)}{\varphi(y1)} = 1\frac{\varphi(loss)}{\varphi(y1)} = 1 ， φ(loss)φ(y2)=−1+x=−3\frac{\varphi(loss)}{\varphi(y2)} = -1 + x = -3\frac{\varphi(loss)}{\varphi(y2)} = -1 + x = -3 ; φ(loss)φ(x)=y2=17\frac{\varphi(loss)}{\varphi(x)} = y2 = 17\frac{\varphi(loss)}{\varphi(x)} = y2 = 17 ; 而 loss_grad =1, 根据公式(5)可以更新：x(-2, 17) ; y1(1, 1) ; y2(17, -3) 。<img src="https://pica.zhimg.com/50/v2-ae1ee0ce2b4891cc499d07026a2f1f88_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="861" data-rawheight="313" data-original-token="v2-ae1ee0ce2b4891cc499d07026a2f1f88" class="origin_image zh-lightbox-thumb" width="861" data-original="https://picx.zhimg.com/v2-ae1ee0ce2b4891cc499d07026a2f1f88_r.jpg?source=2c26e567"/>根据公式（4）可得： φ(y2)φ(z)=2∗z=−8\frac{\varphi(y2)}{\varphi(z)} = 2*z = -8 \frac{\varphi(y2)}{\varphi(z)} = 2*z = -8 , φ(y2)φ(x)=1\frac{\varphi(y2)}{\varphi(x)} = 1\frac{\varphi(y2)}{\varphi(x)} = 1 , 而 y2_grad = -3，则更新：x(-2, 14) ; z(-4, 24) 。根据公式（3）得 φ(y1)φ(z)=1\frac{\varphi(y1)}{\varphi(z)} = 1\frac{\varphi(y1)}{\varphi(z)} = 1 , 而y1_grad = 1，更新：z(-4, 25)。<img src="https://picx.zhimg.com/50/v2-247ad461b2f3d3fc78b77f80f319f458_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="861" data-rawheight="313" data-original-token="v2-247ad461b2f3d3fc78b77f80f319f458" class="origin_image zh-lightbox-thumb" width="861" data-original="https://picx.zhimg.com/v2-247ad461b2f3d3fc78b77f80f319f458_r.jpg?source=2c26e567"/>根据公式（2）得 φ(z)φ(x)=2\frac{\varphi(z)}{\varphi(x)} = 2\frac{\varphi(z)}{\varphi(x)} = 2 ， 而z_grad = 25 更新x(-2, 64)。<img src="https://picx.zhimg.com/50/v2-0ad035a05b402b88ae186c968923f797_720w.jpg?source=2c26e567" data-caption="" data-size="normal" data-rawwidth="861" data-rawheight="313" data-original-token="v2-0ad035a05b402b88ae186c968923f797" class="origin_image zh-lightbox-thumb" width="861" data-original="https://pic1.zhimg.com/v2-0ad035a05b402b88ae186c968923f797_r.jpg?source=2c26e567"/>最后，所有值结果如下： x(-2, 64) ; z(-4, 25); y1(1, 1) ; y2(17, -3) ; loss(-50, 1); 通过附件1代码可验证结果的正确性。3 代码实现通常而言神经网络的基本要素有三个：数据（tensor）、操作/公式（ops）、计算图（graph）。对于tensor的定义主要是：数值（val)、梯度（grad），而求解过程涉及：前向计算公式、反向计算公式和计算流程（graph）。程序实现时，定义好每个tensor之间forward算式、backward算式，以及记录好计算的流程。<img src="https://pica.zhimg.com/50/v2-bf3491d6af93a48ef282e853cbbd631a_720w.jpg?source=2c26e567" data-size="normal" data-rawwidth="1115" data-rawheight="471" data-original-token="v2-0d1ec710a396bcfe3bfcb40a001534ee" class="origin_image zh-lightbox-thumb" width="1115" data-original="https://pic1.zhimg.com/v2-bf3491d6af93a48ef282e853cbbd631a_r.jpg?source=2c26e567"/>x表示输入tensor，y表示输出tensor关键问题：对于程序而言这个计算图的关系存放哪里？在实现中，可以把多维的关系转变为一维存储格式记录下来，这个过程称为拓扑图(topo)构建，如上图中的“3”所示。程序实现关键动作：首先，为每个tensor构建一个cache；这个cache用于存储tensor上下游之间的关系。class Tensor : public enable_shared_from_this<Tensor> {
public:
double val; // 存储value值
double grad; // 存储梯度
void (Tensor::_\_backwardFunc)(); // 记录反向运算的函数
shared_ptr<vector<TensorPtr>> cache; // 记录节点之间的关系
其次，在ops中为cache注册数据；比如一个加法运算ops定义中，输出out的cache里面记录了两个输入值的指针。TensorPtr operator+(TensorPtr tensorA, TensorPtr tensorB) {
shared_ptr<Tensor> out;
out = make_shared<Tensor>();
out->val = tensorA->val + tensorB->val;
out->cache->push_back(tensorA);
out->cache->push_back(tensorB);
out->\_backwardFunc = &Tensor::AddBackward;
return out;
}
最后，完成拓扑排序，构建topo。通过递归遍历的方式，将cache信息放入堆栈中。构建拓扑图的代码片段：  
void BuildTopo(TensorPtr tensor, set<TensorPtr> &u_set,
stack<TensorPtr> &topo) {
if (!u_set.count(tensor)) {
u_set.insert(tensor);
for (auto i : _(tensor->cache)) {
BuildTopo(i, u_set, topo);
}
topo.push(tensor);
}
}
利用topo栈里面记录的tensor，就可以知道反向运算的求解顺序。autograd里面的“auto”指代的是一处tensor的梯度求解能够触发所有相关tensor的梯度求解，topo配合递归完成了这个过程。有了拓扑图，梯度运算就能逐步触发。可以把用例的整个过程用C++代码实现。原理尽在代码中：#include <iostream>
#include <memory>
#include <set>
#include <stack>
#include <vector>

using namespace std;
class Tensor;
using TensorPtr = shared_ptr<Tensor>;

class Tensor : public enable_shared_from_this<Tensor> {
public:
double val; // 存储value值
double grad; // 存储梯度
void (Tensor::\*\_backwardFunc)(); // 记录反向运算的函数
shared_ptr<vector<TensorPtr>> cache; // 记录节点之间的关系

Tensor(double value = 0, double gradient = 0)
: val(value), grad(gradient), \_backwardFunc(NULL),
cache(make_shared<vector<TensorPtr>>()) {}

void AddBackward() {
for (auto i : \*cache) {
i->grad += grad;
}
}

void SubBackward() {
(*cache)[0]->grad += grad;
(*cache)[1]->grad -= grad;
}

void MulBackward() {
(_cache)[0]->grad += grad _ (*cache)[1]->val;
(*cache)[1]->grad += grad * (*cache)[0]->val;
}

void ReluBackward() { AddBackward(); } // relu在大于0时，跟add的反向运算一样

TensorPtr relu() {
shared_ptr<Tensor> out = make_shared<Tensor>();
out->val = this->val > 0 ? this->val : 0;
if (this->val > 0)
out->cache->push_back(shared_from_this());
out->\_backwardFunc = &Tensor::ReluBackward;
return out;
}
// 求解节点之间Topo关系。
// 思路：若节点还没有遍历就放入一个栈中，遍历过了就不放入。
void BuildTopo(TensorPtr tensor, set<TensorPtr> &u_set,
stack<TensorPtr> &topo) {
if (!u_set.count(tensor)) {
u_set.insert(tensor);
for (auto i : \*(tensor->cache)) {
BuildTopo(i, u_set, topo);
}
topo.push(tensor);
}
}

void Backward() {
set<TensorPtr> u_set;
stack<TensorPtr> topo;
this->grad = 1;
BuildTopo(shared_from_this(), u_set, topo); // 求解tope关系
while (!topo.empty()) {
auto tensor = topo.top();
topo.pop();
if (tensor->\_backwardFunc) {
((_tensor)._(tensor->\_backwardFunc))(); // 节点执行反向传播运算
}
}
}
};

TensorPtr operator+(TensorPtr tensorA, TensorPtr tensorB) {
shared_ptr<Tensor> out;
out = make_shared<Tensor>();
out->val = tensorA->val + tensorB->val;
out->cache->push_back(tensorA);
out->cache->push_back(tensorB);
out->\_backwardFunc = &Tensor::AddBackward;
return out;
}

TensorPtr operator+(TensorPtr tensorA, double val) {
TensorPtr out = make_shared<Tensor>(val);
return out + tensorA;
}

TensorPtr operator-(TensorPtr tensorA, TensorPtr tensorB) {
TensorPtr out = make_shared<Tensor>();
out->val = tensorA->val - tensorB->val;
out->cache->push_back(tensorA);
out->cache->push_back(tensorB);
out->\_backwardFunc = &Tensor::SubBackward;
return out;
}

TensorPtr operator-(TensorPtr tensorA, double val) {
TensorPtr out = make_shared<Tensor>(val);
return tensorA - out;
}

TensorPtr operator*(TensorPtr tensorA, TensorPtr tensorB) {
TensorPtr out = make_shared<Tensor>();
out->val = tensorA->val * tensorB->val;
out->cache->push_back(tensorA);
out->cache->push_back(tensorB);
out->\_backwardFunc = &Tensor::MulBackward;
return out;
}

TensorPtr operator*(TensorPtr tensorA, double val) {
TensorPtr out = make_shared<Tensor>(val);
return out * tensorA;
}

TensorPtr operator*(double val, TensorPtr tensorA) {
TensorPtr out = make_shared<Tensor>(val);
return out * tensorA;
}

int main() {
TensorPtr x = make_shared<Tensor>(-2);
TensorPtr z, y1, y2, loss;
z = 2 _ x;
y1 = z + 5;
y2 = (z _ z)->relu() + x + 3;
loss = y1 - y2 + y2 _ x;
loss->Backward();
cout << loss->val << endl;
cout << x->grad << endl;
return 0;
}
程序最后获得的输出与PyTorch的autograd输出保持一致。从原理公式、计算图到代码实现，对autograd有了基本了解。示例中仅展现了一维数据的运算，而对于多维数据的运算还需用J矩阵辅助处理，但是基本原理类似。PyTorch实现还要考虑很多问题，如CPU与GPU运算的差异，感兴趣的读者可以直接尝试阅读源码的逻辑。附1：import torch
x = torch.tensor([-2], dtype=torch.float32, requires_grad=True)
z = 2 _ x
y1 = z + 5
y2 = (z _ z).relu() + x + 3
loss = y1 - y2 + y2 _ x

# 保存中间梯度

z.retain_grad()
y1.retain_grad()
y2.retain_grad()
loss.backward()
print(f"x: {x.item()} x.grad: {x.grad.item()}")
print(f"z: {z.item()} z.grad: {z.grad.item()}")
print(f"y1: {y1.item()} y1.grad: {y1.grad.item()}")
print(f"y2: {y2.item()} y2.grad: {y2.grad.item()}")附2：公式1~5的一般求解算式：Step1正向求解： 代入x值：z=−4,y1=1,y2=17,loss=−50 z = -4, y1 = 1, y2 = 17 , loss = -50 z = -4, y1 = 1, y2 = 17 , loss = -50 Step2反向求导： 计算loss梯度： ∇loss=1∇ loss = 1∇ loss = 1 计算y1的梯度： ∇y1=∂(loss)/∂(y1)=1∇ y1 =\partial(loss)/\partial(y1) = 1 ∇ y1 =\partial(loss)/\partial(y1) = 1 计算y2的梯度： ∇y2=∂(loss)/∂(y2)=−1+x=−3∇ y2 = \partial(loss)/\partial(y2) = -1 + x = -3 ∇ y2 = \partial(loss)/\partial(y2) = -1 + x = -3 计算z的梯度： ∇z=∂(loss)/∂(z)=∂(loss)/∂(y1)⋅∂(y1)/∂(z)+∂(loss)/∂(y2)⋅∂(y2)/∂(z)∇ z = \partial(loss)/\partial(z) = \partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) + \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(z)∇ z = \partial(loss)/\partial(z) = \partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) + \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(z) , =∇y1∗1+∇y2∗2∗z=1∗1+−3∗2∗−4=25= ∇ y1 _ 1 +∇ y2 _ 2*z = 1 * 1 + -3 _ 2 _-4 = 25= ∇ y1 _ 1 +∇ y2 _ 2*z = 1 * 1 + -3 _ 2 _-4 = 25 计算x的梯度： ∇x=∂(loss)/∂(y1)⋅∂(y1)/∂(z)⋅∂(z)/∂(x)+∂(loss)/∂(y2)⋅∂(y2)/∂(x)+y2∇ x = \partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) \cdot \partial(z)/\partial(x) + \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(x) + y2∇ x = \partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) \cdot \partial(z)/\partial(x) + \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(x) + y2 其中： ∂(loss)/∂(y1)⋅∂(y1)/∂(z)⋅∂(z)/∂(x)=1∗1∗2=2\partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) \cdot\partial(z)/\partial(x) = 1 _ 1 _ 2 = 2\partial(loss)/\partial(y1) \cdot \partial(y1)/\partial(z) \cdot\partial(z)/\partial(x) = 1 _ 1 _ 2 = 2, ∂(loss)/∂(y2)⋅∂(y2)/∂(x)=−3∗(−4∗4+1)=45 \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(x) = -3 _ ( -4 _ 4 + 1) = 45 \partial(loss)/\partial(y2) \cdot \partial(y2)/\partial(x) = -3 _ ( -4 _ 4 + 1) = 45 所以： ∇x=2+45+17=64∇ x = 2 + 45 + 17 = 64 ∇ x = 2 + 45 + 17 = 64 基于这个回答写了一个MLP训练，感兴趣可以看看：不用 PyTorch从零实现MLP训练全流程17 赞同 · 8 评论 文章欢迎点赞、关注、留言讨论。 文中不足之处请 @kaiyuan参考^https://pytorch.apachecn.org/docs/1.4/blitz/autograd_tutorial.html
