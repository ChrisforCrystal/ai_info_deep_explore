vLLM Scheduler逻辑难啃？先手搓一个基础调度器
kaiyuan
kaiyuan​
大模型话题下的优秀答主
已关注
收录于 · LLM推理基础与框架
199 人赞同了该文章
​
目录
收起
1 基本逻辑
1.1 问题分析
1.2 设计要求
2 模块实现
2.1 输入与生成
2.2 辅助模块
2.3 Scheduler模块实现
3 测试与可视化
3.1 输出测试
3.2 可视化
Scheduler作为请求调度下发单元，串联着推理框架的各个模块。Scheduler需要支持continuous batching、chunked prefill、kv connector、priority等功能。vLLM的Scheduler经过V0到V1的大版本迭代和若干个小迭代后，其逻辑变得越来越复杂。对于初学者来说，其阅读难度较大。本文从最原始的诉求出发，带读者构建一个最基础的调度器，通过抓住概念的核心，达到事半功倍的效果。

本文提供Scheduler队列的可视化，辅助理解调度器的逻辑：

动图封面
本文代码地址：vllm_basic_scheduler.ipynb[1]

上一篇：

vLLM不知如何开始？看这篇：vLLM框架快速入门引导
537 赞同 · 22 评论 文章
1 基本逻辑
1.1 问题分析
一般情况下，用户请求的抵达时间是不一致的；每个请求生成的字符长度也是不一致的。如下图所示，调度器(Scheduler)分配请求给执行器(Executor)处理的一般流程示例：请求0,1,2已在处理，而请求3,4将陆续抵达。此刻，Scheduler仅下发了请求0,1给Executor运算。

所以Scheduler最先要解决的问题：协调好抵达时间不同、计算长度不一样的用户请求。

进一步，每个请求要消耗KV cache资源，而资源的大小与请求的序列长度成正比。调度器要向KV cache管理模块申请资源(逻辑存储空间)。

Scheduler要解决另一个问题：根据资源的情况决定哪些请求将要执行。

除了这两个基本问题以外，调度器还要解决prefill与decode混合计算、单步下发多步执行、KV cache传输等问题。

为了解决这些问题，Scheduler V1的实现代码已接近2K[2]。在《vLLM框架快速入门引导》中介绍了Scheduler的基本流程。

Scheduler示意图
1.2 设计要求
虽然架构图经过了精简，但仍能看到多组队列协同的逻辑。为了帮助读者快速地了解Scheduler，这里只保留Scheduler的基本功能，删除了中、高阶能力。

支持：Continuous-batching[3]；
支持：prefill和decode的执行优先级控制，默认prefill优先；
支持：decode阶段资源不足时触发抢占，后进入的请求会被排出；
不考虑：prefill和decode混合执行。
不考虑：chunked prefill、kv connector、异步下发、priority等功能。

Continuous-batching（右侧）
decode阶段资源不足触发的抢占功能说明：如下图所示，执行队列(executing queue)里面有三个请求(0~2)、等待队列(waiting queue)里面有两个请求(3和4)。

设定处理序列的总长度不超过30tokens。如下图所示，当请求(0、1、2)总长度超过30tokens时，请求2被排出，并放入waiting队列。

请求(ID 2)被抢占
2 模块实现
要实现端到端的scheduler过程，除了实现scheduler模块本身以外，还要构造token转换、模拟LLM、以及一些辅助模块。

2.1 输入与生成
输入处理：用户请求的prompts转换成token ids；
LLM生成：模拟LLM生成数据。
从prompts到tokens的转换

方式1：使用transformers模块的AutoTokenizer

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
prompts = [
"hi, I'm kaiyuan",
"Do you subscribe InfraTech?",
]
prompts = [
tokenizer.apply_chat_template(
[{"role": "user", "content": prompt}],
tokenize=False,
add_generation_prompt=True,
)
for prompt in prompts
]
for prompt in prompts:
print(tokenizer.encode(prompt))
输出内容：

[151644, 872, 198, 6023, 11, 358, 2776, 595, 2143, 88, 10386, 151645, 198, 151644, 77091, 198]
[151644, 872, 198, 5404, 498, 17963, 14921, 956, 34097, 30, 151645, 198, 151644, 77091, 198]
方式2： 构建一个随机整数输出来充当tokens。

可以采用随机数的方式，每次生成一个token。

array*length = np.random.randint(min_len, max_len + 1)
random_array = [np.random.randint(min_val, max_val + 1) for * in range(array_length)]
构造一个模拟的LLM

def run_fake_model(seqs, max_len: int = 15, min_val: int = -1, max_val: int = 999, eos: int=-1):
token_ids = [eos if len(seq) >= max_len else np.random.randint(min_val, max_val + 1) for seq in seqs]
return token_ids
2.2 辅助模块
1、用Sequence类承载单个请求的相关信息，如：序列长度、请求状态、KV cache blocks的使用情况等。

class Sequence:
block_size = 256
counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

2、KV cache管理模块。 设计要求：

用paged attention的原理来管理KV cache。
以Block为单位分配KV cache，每个Block包含若干tokens cache空间。
Blocks统一由BlockManager管理。
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

2.3 Scheduler模块实现
Scheduler模块主要通过两个队列(waiting、running)实现请求的轮转，prefill阶段和decode阶段的处理逻辑不一样。代码如下：

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_seqs = 0
        self.num_batched_tokens = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def prefill(self):
        scheduled_seqs = []
        while self.waiting and self.num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if self.num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # 资源不足时打断
            self.num_seqs += 1
            self.block_manager.allocate(seq)
            self.num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs, True

    def decode(self):
        scheduled_seqs = []
        while self.running and self.num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
          self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占：
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 调度函数
    def schedule(self, prefill_first=True) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        self.num_seqs = 0
        self.num_batched_tokens = 0
        is_prefill = True

        if prefill_first:
            first_call, second_call = self.prefill, self.decode
        else:
            first_call, second_call = self.decode, self.prefill

        scheduled_seqs, is_prefill = first_call()
        if scheduled_seqs:
            return scheduled_seqs, is_prefill

        scheduled_seqs, is_prefill = second_call()
        if scheduled_seqs:
          return scheduled_seqs, is_prefill
        assert scheduled_seqs # 当没有任何请求被处理时，报错。 资源不足可能出现此情况。

    # 后处理：
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

除此之外还要构建config、采样参数等，具体参考vllm_basic_scheduler.ipynb第2节。

3 测试与可视化
3.1 输出测试

# 创建一个调度器：

scheduler = Scheduler(config)

# 添加请求示例：

token_ids = tokenizer.encode("hi, I'm kaiyuan")
sampling_params = SamplingParams()
seq = Sequence(token_ids, sampling_params)
scheduler.add(seq)
添加2个请求到scheduler中：

# 打印输入请求情况

print("scheduler waiting queue: ")
for id, seq in enumerate(scheduler.waiting):
print(f"id:{id} seq:{seq.token_ids}")

# 打印输出：

"""
scheduler waiting queue:
id:0 seq:[5404, 498, 17963, 14921, 956, 34097, 30]
id:1 seq:[6023, 11, 358, 2776, 595, 2143, 88, 10386]
"""
构建一个请求处理循环：

while not scheduler.is_finished():
seqs, is_prefill = scheduler.schedule()
token_ids = run_fake_model(seqs, config.max_model_len)
scheduler.postprocess(seqs, token_ids)
for id, seq in enumerate(seqs):
print(f"id:{id} seq:{seq.token_ids}")
得到如下的输出，可以看到请求的字符逐步生成，直到输出完成。

数字-1表示结束
3.2 可视化
为了让Scheduler的队列可读性更强，用matplotlib将scheduler的两个队列打印出来，可视化代码参考notebook示例的第4节，这里不做展开。

队列内容打印
注意：执行队列(EXECUTING QUEUE)表示scheduler的schedule函数的输出内容。等待队列(WAITING QUEUE)表示scheduler的waiting、running队列里面未执行的请求。

示例1：prefill优先调度

参数：

max_num_seqs=3，最多运行3个请求；
max_model_len=15，每个请求长度最大值为15；
init_reqs=5，初始化时请求数量为5；
totoal_reqs=8，总请求数为8；
add_reqs_step=15，在第15次forward的时候加入剩余请求。
输出如下：

动图封面
主要观测第19帧时的处理动作：19帧处有新请求抵达，这些新请求会优先运行。

prefill优先执行
示例2：资源不足抢占

参数：num_kvcache_blocks=3，kvcache_block_size=10

总资源数量为3\*10，超过该数值时发生资源抢占：会将执行队列的队尾请求转移到等待队列。

动图封面
第9帧时触发抢占
示例3：decode优先调度

decoding运算优先执行。可以看到这种方式下，当执行队列有请求在处理时，等待队列的请求无法进入执行队列。

动图封面
由于不具备prefill、decode混合执行的功能，执行队列里面要么只有prefill请求，要么只有decode请求。

本文仅帮助初学者理解基础的Scheduler能力，深入学习Scheduler可参考：
