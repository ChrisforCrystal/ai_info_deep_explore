import copy
import numpy as np
from enum import Enum
from itertools import count
from collections import deque

# ==========================================
# 核心数据结构定义
# ==========================================

class SequenceStatus(Enum):
    WAITING = 1   # 在等位区排队（还没算过）
    RUNNING = 2   # 正在就餐区吃（GPU中正在算）
    FINISHED = 3  # 吃完买单了（生成结束）

class SamplingParams:
    """顾客的食量要求"""
    def __init__(self, temperature=1.0, max_tokens=15, ignore_eos=False):
        self.max_tokens = max_tokens  # 最多长到多大
        self.ignore_eos = ignore_eos

class Sequence:
    """顾客（一个大模型请求）"""
    counter = count() # 全局发号器

    def __init__(self, token_ids: list[int], sampling_params=None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        self.seq_id = next(Sequence.counter)  # 顾客拿到的排队流水号
        self.status = SequenceStatus.WAITING  # 初始都在门外等
        self.token_ids = copy.copy(token_ids) # 顾客刚进门自带的身材（Prompt）
        
        self.num_tokens = len(self.token_ids) # 当前这句话总共多长了
        self.num_cached_tokens = 0            # 已经分配了多少个“物理椅子（KV Cache）”
        self.block_table = []                 # 记录该顾客屁股底下的所有椅子编号
        
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.num_completion_tokens = 0        # 记录他在这里新吃了多少口（生成的Token数）

    def append_token(self, token_id):
        """【吃下一口】：模型帮他又生成了一个新词，体格变大了一点"""
        self.token_ids.append(token_id)
        self.num_tokens += 1
        self.num_completion_tokens += 1

    def __len__(self):
        return self.num_tokens


class Block:
    """一把固定的长排椅（对应物理显存中的一个 Block，通常装16个Tokens）"""
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0  # 0代表没人坐，是空闲的；1代表有人坐
        self.token_ids = []

    def reset(self):
        """服务员收桌子"""
        self.ref_count = 0
        self.token_ids = []

# ==========================================
# 内存管理系统 (底层大管家)
# ==========================================

class BlockManager:
    """KV Cache的的大管家，管着店里所有的板凳"""
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size        # 一把排椅能挤下几个人
        self.blocks = [Block(i) for i in range(num_blocks)] # 全店就这么多固定椅子
        self.free_blocks = list(self.blocks) # 刚开门，全是空的

    def can_allocate(self, seq: Sequence) -> bool:
        """【迎宾探测】：看看仓库里的空椅子，够不够装下这个刚进门的胖顾客"""
        num_blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_blocks_needed

    def allocate(self, seq: Sequence):
        """【入座】：给刚进门的顾客安排一套初始的椅子"""
        num_blocks_needed = (seq.num_tokens + self.block_size - 1) // self.block_size
        for _ in range(num_blocks_needed):
            block = self.free_blocks.pop()
            block.ref_count = 1
            seq.block_table.append(block) # 记在顾客名下
        seq.num_cached_tokens = len(seq.block_table) * self.block_size

    def can_append(self, seq: Sequence) -> bool:
        """【生长期探测】：顾客每次生成新词会变胖，看看之前分给他的排椅是不是坐满了，如果满了，库房还有没有新椅子给他？"""
        if seq.num_tokens <= seq.num_cached_tokens:
            return True # 旧椅子还挤得下
        return len(self.free_blocks) > 0 # 旧椅子满了，就看库房还有空的不

    def may_append(self, seq: Sequence):
        """【加椅子】：老椅子挤满了，从库房拽一把新排椅接到他屁股后面"""
        if seq.num_tokens > seq.num_cached_tokens:
            block = self.free_blocks.pop()
            block.ref_count = 1
            seq.block_table.append(block)
            seq.num_cached_tokens += self.block_size

    def deallocate(self, seq: Sequence):
        """【结账走人】：回收这个顾客身上挂着的所有椅子"""
        for block in seq.block_table:
            block.reset()
            self.free_blocks.append(block)
        seq.block_table = []
        seq.num_cached_tokens = 0


# ==========================================
# 调度系统系统 (大堂经理)
# ==========================================

class Config:
    def __init__(self):
        self.max_num_seqs = 4                 # 并发上限：最多让几桌人一起吃
        self.max_num_batched_tokens = 50      # CUDA运算上限：这一批处理合起来最多多少Token
        self.num_kvcache_blocks = 10          # 显存容量：全店只有10把排椅
        self.kvcache_block_size = 5           # 排椅尺寸：1把排椅能存5个Token
        self.eos = -1                         # 句号/结束符标记
        self.max_model_len = 15               # 单人生存上限


class Scheduler:
    """大堂经理 Scheduler"""
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        self.waiting: deque[Sequence] = deque() # 【等位区】
        self.running: deque[Sequence] = deque() # 【就餐区】
        self.num_seqs = 0
        self.num_batched_tokens = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """接待！给个排号条去等位区呆着"""
        self.waiting.append(seq)

    def prefill(self):
        """【优先：迎宾入座 (Prefill阶段)】"""
        scheduled_seqs = []
        # 等位区有人，且店里没满员
        while self.waiting and self.num_seqs < self.max_num_seqs:
            seq = self.waiting[0] # 看最前面那位
            
            # 卡点：算力够不够？ 椅子够不够拿？
            if self.num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # 搞不定了，停止迎新的客人
            
            self.num_seqs += 1
            self.block_manager.allocate(seq) # 发放初始椅子 (分配 KV Cache)
            self.num_batched_tokens += len(seq) - seq.num_cached_tokens # 登记他的身材数据给后台(GPU)算力
            
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()     # 移出等位区
            self.running.append(seq)   # 送进就餐区
            scheduled_seqs.append(seq)
        return scheduled_seqs, True # True 表示这是在新客 Prefill 逻辑

    def decode(self):
        """【其次：服务场内老客 (Decode阶段)】"""
        scheduled_seqs = []
        while self.running and self.num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 危机检测：此客体量增长，看看还有没有后续的空椅子给他
            while not self.block_manager.can_append(seq):
                # OOM危机！全店椅子耗尽 🚨
                if self.running:
                    # 【核心保命机制：抢占 Preempt】
                    # 把场内排在最后面的那位倒霉蛋客人拎起来踢出去
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    # 场内就他自己了还没资源，连自己也踢了
                    self.preempt(seq)
                    break 
            else:
                 # 椅子够，正常服务他
                self.num_seqs += 1
                self.block_manager.may_append(seq) # 需要加椅子的自动加椅子
                self.num_batched_tokens += 1       # Decode每步只蹦出1个字，所以固定加1
                scheduled_seqs.append(seq)
                
        # 轮询服务完一次，把正在就餐的这批人塞回原地
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False # False 表示这是在老客 Decode 逻辑

    def preempt(self, seq: Sequence):
        """【踢人法则：暴力夺椅】"""
        print(f"⚠️ 警告: 资源枯竭！请求 No.{seq.seq_id} 被无情踢回等位区重算！")
        seq.status = SequenceStatus.WAITING  # 降级！
        self.block_manager.deallocate(seq)   # 回收！！
        self.waiting.appendleft(seq)         # 给点补偿，塞到等位区第一名

    def schedule(self, prefill_first=True):
        """【大堂经理的心跳引擎（主分发口）】"""
        scheduled_seqs = []
        self.num_seqs = 0
        self.num_batched_tokens = 0
        
        # 决定是优先接新客，还是优先喂老客
        first_call = self.prefill if prefill_first else self.decode
        second_call = self.decode if prefill_first else self.prefill

        # 第一志愿
        scheduled_seqs, is_prefill = first_call()
        if scheduled_seqs:
            return scheduled_seqs, is_prefill

        # 第二志愿 (通常是新客因为满了进不来，就转去喂老客)
        scheduled_seqs, is_prefill = second_call()
        if scheduled_seqs:
            return scheduled_seqs, is_prefill
            
        return [], False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """【吃完结账】：接受底层传上来的计算结果，判断是不是吃饱了"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            
            # 看看碰到结束符没有？限制长度到了没有？
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq) # 撒手！释放所有显存 Block
                self.running.remove(seq)           # 让位！

# ==========================================
# 大模型引擎循环跑马场 (Moc 演示) 
# ==========================================

def run_fake_model(seqs, config):
    """一个超级智障的大模型模拟器，随机蹦数字代表Token"""
    return [config.eos if len(seq)>=config.max_model_len else np.random.randint(10, 99) for seq in seqs]

if __name__ == "__main__":
    print("🚀 vLLM 调度器模拟启动！")
    config = Config()
    scheduler = Scheduler(config)

    # 模拟接到了两个HTTP请求，带了长短不一的句子
    scheduler.add(Sequence([101, 102, 103])) # 短请求
    scheduler.add(Sequence([201, 202, 203, 204, 205, 206, 207])) # 长请求
    
    step = 0
    while not scheduler.is_finished():
        step += 1
        print(f"\n--- 第 {step} 步 ---")
        
        # 1. 大堂经理做调度决定 (要么放新客，要么喂老客)
        seqs, is_prefill = scheduler.schedule(prefill_first=True)
        if not seqs:
            print("宕机！死锁了！")
            break
            
        phase = "【入座 Prefill】" if is_prefill else "【生成 Decode】"
        print(f"当前阶段: {phase}, 被调度的顾客IDs: {[s.seq_id for s in seqs]}")
        
        # 2. 扔给GPU(假模型)算力，把这批词算出来
        out_tokens = run_fake_model(seqs, config)
        
        # 3. 把新算出来的词挂在这些人身上，看谁吃完下桌
        scheduler.postprocess(seqs, out_tokens)

        # 检查场内状况
        for s in seqs:
            print(f"顾客号 {s.seq_id} 当前体格:{s.token_ids} | 占椅量:{len(s.block_table)}把")
            if s.status == SequenceStatus.FINISHED:
                print(f"✅ 顾客号 {s.seq_id} 圆满吃撑，买单走人！")
