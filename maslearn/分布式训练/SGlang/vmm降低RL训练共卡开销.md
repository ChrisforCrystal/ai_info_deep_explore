降低RL训推共卡开销：SGLang/vLLM的无缝切换实现与分析
kaiyuan
kaiyuan​
大模型话题下的优秀答主
已关注
收录于 · LLM推理基础与框架
航海家 曹宇 等 73 人赞同
​
目录
收起
1 问题背景
2 推理框架的启动耗时
3 CUDA Graph的重用
3.1 虚拟地址介绍
3.2 可行性验证
3.3 方案的实施
4 vLLM框架
4.1 权重热更新
4.2 睡眠模式介绍
4.3 训推切换测试
4.4 数据分析
5 SGLang框架
5.1 权重热更新
5.2 睡眠模式
在大模型后训练场景中，不同阶段的模块需要分时复用GPU硬件资源。在共卡（co-location）模式下，任务（如训练、推理、生成）之间的切换会引入额外开销，进而降低强化学习（RL）的整体迭代效率。目前，生成（rollout）阶段常借助SGLang、vLLM等推理框架完成，因此，如何降低这些框架的启动成本、释放充足显存，并实现权重的快速传递，已成为提升rollout阶段效率的关键。本文围绕推理框架的切换问题展开讨论，主要涉及以下几个方向：

框架启动耗时的构成，以及何为“预热启动”；
如何更充分地释放显存；
如何实现框架权重的在线更新；
何为“睡眠模式”及其使用方法。
1 问题背景
场景1：RL训推共卡(co-location)。在RL的后训练过程中，一般会有多个阶段[1]：1生成(Generation)、2准备/推理(Preparation/Inference)、3训练(Training)，这些阶段是串行执行。为了提升资源利用效率，通常会安排不同阶段的任务共用GPU设备。例如，在生成(Generation)阶段结束后，推理任务所占用的资源会被释放，以便用于下一阶段的计算。

场景2：推理的模型切换。在AgentiRL的训练中，同一批GPU资源被多个模型分时复用。

这两个场景面临相同的问题：尽管任务是串行执行、计算资源的使用并无冲突，但由于显存空间有限，任务间的上下文切换仍会带来一定的开销。对于参数量较大的模型而言，这类切换时间不容忽视。因此，如何降低框架再次启动的开销，成为本文讨论的重点。

2 推理框架的启动耗时
在推理服务的部署中，框架的启动耗时通常并非关注重点，因为服务一般会在完全就绪（ready）后才开始接收客户端请求。然而在强化学习（RL）场景中，框架需要频繁启动和停止，且启动时间与模型大小正相关。为清晰说明这一影响，本文将通过简单示例，展示离线模式下框架的启动耗时。

# -_- coding: gbk -_-

import torch
import time
import logging

import sglang as sgl
logging.basicConfig(level=logging.INFO)

def main(): # Sample prompts.
prompts = [
"Hello, I'm kaiyuan",
"Do you subscribe InfraTech?",
]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 30}

    # Create an LLM.
    t0 = time.time()
    llm = sgl.Engine(
        model_path="./Qwen2.5-7B-Instruct",
        tp_size=1,
        cuda_graph_max_bs=8,
    )
    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()

    print(f"Start time: {t1- t0}s . Generate time: {t2 - t1}s")
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if **name** == "**main**":
main()
输出示例：

Start time: 24.483609199523926s . Generate time: 0.9596457481384277s
通过对比可以看出，相比推理任务本身的生成时间，框架启动的耗时相对较长。启动过程通常涉及以下环节：

Python端的上下文准备；
GPU端的初始化与上下文准备；
显存的分配与创建；
GPU算子的即时（JIT）编译；
GPU CUDA Graph的捕获（若开启图模式）。
其中，算子（如FlashInfer/FlashAttn）的编译以及CUDA Graph的捕获是启动过程的主要耗时环节。同时，从框架的显存分析[2]中可知，模型权重与KV Cache是显存占用的主要部分。

自然而然地，我们会思考在框架切换出去时，能否只释放KV Cache和模型权重，而保留算子JIT编译的缓存、上下文信息以及CUDA Graph信息？这样，在下一次框架启动时，只需重新分配KV Cache并恢复模型权重即可。通过设计这样一种热启动机制，可以显著缩短框架的启动时间。

但我们都知道在使用显存时，分配的显存地址是驱动根据当前条件进行分配的，同样的动作每次获得的物理地址会发生变化。当物理地址变化时，CUDA Graph需要重新编译。所以问题转化为：能否保证权重创建，KV cache等使用相同的物理地址？答案是：利用CUDA虚拟地址管理VMM[3]（Virtual Memory Management）。

3 CUDA Graph的重用
3.1 虚拟地址介绍
虚拟地址管理是在CUDA物理地址与应用层之间加入的一层地址管理机制，最初主要用于解决显存碎片问题。它将离散的物理块（physical chunk）映射到连续的虚拟地址上，使得上层应用通过与虚拟地址交互，能够获得更大的连续显存空间

在虚拟地址管理中，一个关键特点是：VMM实例的虚拟地址与物理地址之间没有必然的耦合关系。释放physical chunk后，虚拟地址可以保留，同时该物理块仍可被其他应用使用。BasicCUDA[4]中提供了一段关于VMM基本使用的介绍：

https://github.com/CalvinXKY/BasicCUDA/tree/master/memory_opt/vmmgithub.com/CalvinXKY/BasicCUDA/tree/master/memory_opt/vmm
github.com/CalvinXKY/BasicCUDA/tree/master/memory_opt/vmm
更进一步，如果CUDA Graph捕获的是虚拟地址，并且这些虚拟地址保持不变，那么在再次启动推理框架时，只需将新的physical chunk重新映射到原有的虚拟地址上即可。

3.2 可行性验证
可以设计一个实验来验证该方案的可行性，思路如下：

如果CUDA Graph捕获的是虚拟地址，并且这些虚拟地址在过程中保持不变，那么当推理框架再次启动时，只需将新的物理块重新映射到原有的虚拟地址上即可。

具体做法是：通过VMM API来分配和管理显存，依次执行两个矩阵乘法运算，两次运算使用相同的虚拟地址；但在第一个矩阵计算完成后，其对应的物理地址会被释放并重新申请。整个过程由CUDA Graph捕获，之后可检查该CUDA Graph是否因物理地址变化而需要重新编译。

主要执行步骤
判断图是否需要重编译的关键函数：

cudaError_t update_status = cudaGraphExecUpdate(graph_exec, new_graph, &error_node, &update_result);
完整代码地址：vmm/virtual_mem_with_cuda_graph.cu[5]，按照readme步骤完成编译打印，得到输出如下（节选）：

Capturing new graph with updated data...
New graph captured successfully

Checking if existing graph_exec can be updated with new graph...
Update check completed in 5us
Update result: Success (code: 0)

cudaGraphExecUpdate SUCCESS!
Meaning: The existing graph_exec can handle the new data without recompilation.
Reason: Only memory content changed, addresses remain identical.

Reusing existing graph_exec with new data...
Graph reuse execution time: 36us

Verifying results with new data...
Phase 2: Results are correct

Data change verification:
Phase 1 result sum (first 10): 41880
Phase 2 result sum (first 10): 365409
Difference: 323529
Confirmed: Computation used new dataset
Graph successfully reused: YES
从日志可见输出信息“Graph successfully reused: YES”，表明该CUDA Graph已被成功复用，无需重新编译。

3.3 方案的实施
由于在训练/推理框架中通常使用标准的CUDA API进行显存管理，因此最直接的实现方式是构建一种新的管理机制来替换原有的使用方式。具体而言，可以在原有显存管理方案的基础上，新增一套通过环境变量控制的管理机制。类似的做法在PyTorch的Expandable Segments中已有介绍。

详见PyTorch显存管理介绍与源码解析（三）[6]。

此外，也可采用VMM替代原有的显存管理方式[7]，通过LD_PRELOAD拦截并替换原有的CUDA API[8]来实现。在VeRL/SGLang中，显存管理已按此方式调整，具体机制如下：

创建与使用

创建CUmemGenericAllocationHandle（CUDA内存通用分配句柄），并使用cuMemCreate函数分配物理内存。该句柄包含了待分配内存的属性信息，例如内存的物理位置（如在哪个GPU上）或应提供何种类型的可共享句柄。
使用cuMemAddressReserve保留一段虚拟地址范围。
通过cuMemMap将物理内存映射到该虚拟地址。
将虚拟内存指针和物理内存句柄存储在一个元数据映射表（Metadata Map）中。

物理显存的释放

使用cuMemUnmap将内存从虚拟地址范围中解除映射。
从元数据映射表中获取物理内存句柄，并使用cuMemRelease将其释放。此操作会释放物理内存，但同时保留虚拟地址。

数据恢复（重新绑定物理内存）

创建新的物理内存句柄CUmemGenericAllocationHandle，并使用cuMemCreate函数进行初始化。
使用cuMemAlloc分配物理内存。
通过cuMemMap将新分配的物理内存映射到已存储的虚拟地址。使用新的内存句柄更新元数据映射表。

SGLang与VeRL应用此方式的案例：

使用LD_PRELOAD的优势在于适配速度较快，但其缺点也较为明显：必须确保被劫持的so库与实际运行的so库版本一致，同时需注意系统中没有其他操作也设置了LD_PRELOAD路径而导致冲突。

4 vLLM框架
4.1 权重热更新
在强化学习训练与推理切换时，训练得到的新权重需同步至推理框架。vLLM默认从固定位置读取权重文件，若不计加载效率，可由训练框架（如Megatron）将新权重保存至磁盘并覆盖旧文件，再让vLLM重新加载。另一种更为常见的做法是权重直传（亦称热更新），这也是当前多数RL框架采用的方案，其优势在于能显著缩短框架启动时间。

当然，权重的热更新涉及诸多复杂问题，例如TP异构处理、通信链路选择、通信与计算重叠、异步传输等。此处我们仅聚焦于vLLM框架如何接收新权重这一环节。实现方式上，既可通过Monkey Patch进行修改，也可构造一个自定义的加载器。目前来看，后者的实现方式更为优雅。

在vLLM中，权重加载默认由ModelLoader完成。现有多种实现具体加载逻辑的ModelLoader，它们均继承自位于vllm/model_executor/model_loader/base_loader.py的BaseModelLoader基类。

要实现权重热传递，可通过继承BaseModelLoader并实现自定义逻辑来完成。下面给出一个基本示例：

@register_model_loader("my_loader")
class MyModelLoader(BaseModelLoader):
"""Model loader that will set model weights to random values."""
load_count = 0
def **init**(self, load_config: LoadConfig):
super().**init**(load_config)
if load_config.model_loader_extra_config:
raise ValueError(
f"Model loader extra config is not supported for "
f"load format {load_config.load_format}"
)

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:

        initialize_dummy_weights(model)
        MyModelLoader.load_count += 1
        print(f"==***== load_weights count: {MyModelLoader.load_count} ==***==")

关键在于重写load_weights函数，可通过NVLink、RDMA等方式接收数据并更新权重。示例代码中添加了 print语句，用于验证自定义ModelLoader是否被成功调用。

在LLM中，可通过load_format参数选择自定义的权重加载器：

    llm = LLM(model=model_name,
              dtype='float16',
              load_format='my_loader',
              enable_sleep_mode=True)

4.2 睡眠模式介绍
在vLLM中通过睡眠模式（sleep mode）[9]来控制模型权重、KV cache的显存释放。目前设置了两个等级：

level 1
操作：卸载模型权重并清除KV cache。
内存：模型权重被移动到CPU内存；KV cache被清除。

level 2
操作：同时丢弃模型权重和KV缓存。
内存：模型权重和KV cache都被清除。
在RL应用中一般使用level2模式。调用示例：

# Sleep level 2

# Put the engine to sleep (level=2: discard both weights and KV cache)

llm.sleep(level=2)

# Reallocate weights memory only

llm.wake_up(tags=["weights"])

# Load weights in-place

llm.collective_rpc("reload_weights")

# Reallocate KV cache

llm.wake_up(tags=["kv_cache"])
过程步骤示意如下：

在sleep和wake_up操作中，会分别调用unmap_and_release(handle)与create_and_map(handle)函数，这两个函数定义于vllm/csrc/cumem_allocator.cpp中。

// 代码节选
void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
#ifndef USE_ROCM
CUmemGenericAllocationHandle* p_memHandle) {
#else
CUmemGenericAllocationHandle\*\* p_memHandle,
unsigned long long* chunk_sizes, size_t num_chunks) {
#endif
ensure_context(device);
// Define memory allocation properties
CUmemAllocationProp prop = {};

// Allocate memory using cuMemCreate
CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
if (error_code != 0) {
return;
}
CUDA_CHECK(cuMemMap(d_mem, size, 0, \*p_memHandle, 0));
if (error_code != 0) {
return;
}

CUDA_CHECK(cuMemSetAccess(d_mem, size, &accessDesc, 1));
if (error_code != 0) {
return;
}
在睡眠模式下，使用VMM可以确保唤醒后CUDA Graph无需重新编译，从而显著提升框架的启动速度。睡眠模式与冷启动的对比[10]：

4.3 训推切换测试
设计一个简单的测试用例，用于验证vLLM在睡眠模式下的显存变化，并检查vLLM休眠后其他API能否正常申请和使用显存。

训练阶段显存占用的模拟方法：使用 PyTorch 创建一个超大张量，测试vLLM释放的显存是否可被正常使用。

案例思路：

自定义ModelLoader，以改变vLLM框架中权重的加载行为；
分别收集模型在sleep mode与wake up状态下的显存使用信息；
当vLLM模型处于sleep mode时，使用PyTorch创建张量来模拟训练任务占用显存；
将记录的显存信息打印输出。执行步骤：

创建LLM实例，并执行离线推理；
卸载该LLM的KV Cache和模型权重；
创建一个超大的Tensor，以模拟训练阶段占用显存；
卸载该超大Tensor，模拟训练任务结束；
重新加载LLM的权重与KV Cache；
再次触发LLM推理过程。
最后，观察并分析过程中的日志输出，打印并比对显存在各步骤中的变化情况。实现代码节选：

    outputs = llm.generate(prompts, sampling_params)
    monitor.print_memory("first generated")
    llm.sleep(level=2)
    monitor.print_memory("after sleep")

    # 用torch 尝试占用显存
    elements = int(20 * (1024 ** 3) / 4)
    big_tensor = torch.randn(elements, dtype=torch.float32, device='cuda')

    monitor.print_memory("created big tensor")
    del big_tensor
    gc.collect()
    torch.cuda.empty_cache()

    monitor.print_memory("deleted big tensor")
    # Reallocate weights memory only
    llm.wake_up(tags=["weights"])
    # Load weights in-place
    llm.collective_rpc("reload_weights")
    monitor.print_memory("weights reloaded.")
    # Reallocate KV cache
    llm.wake_up(tags=["kv_cache"])

    monitor.print_memory("kv cache waked up.")
    outputs = llm.generate(prompts, sampling_params)
    monitor.print_memory("second generated")
    print(monitor.history_used)

完整代码参考：InfraTech/llm_infer/switch_role_update_weights.ipynb[11]

运行该用例后，可从日志输出中观察到：在使用虚拟地址管理的情况下，PyTorch所打印的内存信息存在误差，我们以NVIDIA-SMI打印为准。

# NVIDIA-SMI 状态： [GPU 0] 显存: 21.8/80.0 GB (27%) Info：created big tensor

PyTorch CUDA 内存状态查询:
GPU：0 (NVIDIA A100-SXM4-80GB):
已分配: 89.96 GB
缓存保留: 90.21 GB
总内存: 79.25 GB
空闲内存: -10.71 GB
过程内存的变化如下：

阶段介绍：

begin status：起始状态。
first generated：vLLM启动，并完成第一次推理；
after sleep：开启睡眠模式；
created big tensor：用pytorch创建一个20G左右的tensor；
deleted big tensor：删除上一步的tensor，并清除cache；
weights reloaded：重载权重；
kv cache waked up: 创建kv cache显存空间；
second generated：vLLM第二次推理。
4.4 数据分析
为对比睡眠模式启动与冷启动在速度上的差异，实验设置了包括模型类型、睡眠模式等在内的消融变量（详细信息可参考[10]）。以下列举几组实测数据：

数据1：

测试条件：GPU: A100 | vLLM 0.11.0 | Sleep Level: 1 | Compilation:cudagraph_mode: FULL_AND_PIECEWISE

数据2：

测试条件：GPU: A4000 (TP=1) | vLLM 0.11.0 | Sleep Level: 1 | Compilation:cudagraph_mode: FULL_AND_PIECEWISE

数据3：

测试条件：GPU: A100 (TP=1) | vLLM 0.11.0 | Sleep Level: 2 | Compilation:cudagraph_mode: FULL_AND_PIECEWISE

测试数据表明，睡眠模式能显著加快框架启动速度。

5 SGLang框架
在强化学习相关特性支持方面，SGLang支持的功能更多，其对睡眠模式、权重在线传输、确定性计算、负载均衡等[12]功能的支持速度更快。

5.1 权重热更新
SGLang目前已直接支持“从张量更新权重”（Update Weights from Tensor）功能，训练框架在权重更新后可直接将张量传递给SGLang，从而显著提升权重同步的速度。API调用示例如下所示

        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(
            [
                (write_param_name, new_tensor.clone())
                for write_param_name in write_param_names
            ],
            load_format="direct",
        )

        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()

SGLang在Online模式下同样支持权重更新。其接口为POST /update_weights_from_tensor，能够在服务运行中在线更新权重，保持了服务端（Server）形态的运维优势。目前该方式的使用存在以下要求和限制：

此策略要求训练过程与推演引擎能够共享对张量的访问权限。在共置部署中，模型必须始终驻留在GPU上；若将张量移至CPU，则会中断更新路径。对于需要高性能的混合专家模型（MoE）或专用注意力内核而言，与解耦部署的rollouts相比，共卡部署可能会限制某些优化手段的实现。
5.2 睡眠模式
SGLang的睡眠模式与vLLM的功能相似，其主要目的同样是减少启动阶段的时间开销，从而提升强化学习（RL）的整体迭代速度。在SGLang中，该模式分为两个步骤：释放显存（Release Memory）与恢复显存（Resume Memory），并通过标签（tag）控制权重与KV Cache的释放与恢复。

下面给出一个睡眠模式的简单示例，关键步骤如下：

创建一个推理引擎（engine）；
卸载其KV Cache和权重；
加载一组新权重；
恢复引擎的权重；
执行权重更新；
恢复KV Cache，并确认推理功能是否正常。
示意代码如下：

engine = Engine(
model_path=model_path,
random_seed=random_seed,
mem_fraction_static=mem_fraction_static,
base_gpu_id=base_gpu_id,
enable_memory_saver=True,
tp_size=self.\_tp_size,
node_rank=node_rank,
)

# 1 - release kv cache

engine.release_memory_occupation(tags=["kv_cache"])

# 2 - release sglang weights

engine.release_memory_occupation(tags=["weights"])

# 3 - load hf model

hf_model = AutoModelForCausalLM.from_pretrained(
DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
torch_dtype="bfloat16",
device_map=f"cuda:{rank}",
trust_remote_code=True,
).cuda()

# 4 - resume sglang weights and update the weights

engine.resume_memory_occupation(tags=["weights"])
engine.update_weights_from_tensor(
named_tensors=list(hf_model.named_parameters())
)

# 5 - release hf model

del hf_model
hf_model = None
torch.cuda.empty_cache()
time.sleep(3)
torch.cuda.empty_cache()

# 6 - resume slgang kv cache

engine.resume_memory_occupation(tags=["kv_cache"])

# 7 - test llm

outputs = llm.generate(prompts, sampling_params)
权重热更新、睡眠模式能显著加快框架启动速度，并大幅减少RL任务切换的时间开销，在同步、异步模式下都能发挥重要作用。
