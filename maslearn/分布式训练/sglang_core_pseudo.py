import numpy as np
from collections import deque
from enum import Enum

# ==========================================
# 绝活一：Radix Cache (基数树缓存 / 前缀缓存)
# ==========================================

class RadixNode:
    """Radix树的一个节点，代表一段共享的Prompt/Token序列"""
    def __init__(self, tokens: list[int]):
        self.tokens = tokens          # 这段节点存的具体Token序列，比如 [101, 102, 103, 104]
        self.kv_cache_blocks = []     # 对应的物理GPU显存块 (KV Cache)
        self.children = {}            # 子节点：接着这段话往后延伸的不同分支
        self.ref_count = 0            # 有多少个当前的请求正在使用这段节点（为0时可以被淘汰回收）

class RadixCache:
    """SGLang的核心资产：全局前缀缓存树"""
    def __init__(self):
        # 根节点为空，所有Prompt都从这里出发查找
        self.root = RadixNode([])

    def match_prefix(self, token_ids: list[int]):
        """
        【前缀匹配】：拿着新来的Prompt去树里疯狂找同款
        返回：(匹配到的Token数量, 对应的物理KV Cache)
        """
        node = self.root
        match_len = 0
        matched_kv_blocks = []
        
        # 沿着树往下爬，能抠几个词抠几个词
        while match_len < len(token_ids):
            next_token = token_ids[match_len]
            if next_token in node.children:
                # 找到匹配的分支，走进去！
                node = node.children[next_token]
                # 假设这个节点正好存了接下来的这段Token (简单起见每次匹配1个)
                match_len += len(node.tokens)
                matched_kv_blocks.extend(node.kv_cache_blocks)
                node.ref_count += 1
            else:
                break # 树里没有现成的了，匹配到此为止！
                
        return match_len, matched_kv_blocks

    def insert(self, token_ids: list[int], kv_blocks: list):
        """【缓存沉淀】：把辛辛苦苦算出来的新KV Cache种到树上，惠及后人"""
        # (伪代码省略了复杂的树分裂和节点拆分逻辑，仅展示思想)
        pass


# ==========================================
# 基础结构定义
# ==========================================

class SequenceStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3

class Sequence:
    """一个发给SGLang的大模型请求"""
    def __init__(self, seq_id: int, token_ids: list[int], max_tokens: int = 20):
        self.seq_id = seq_id
        self.prompt_tokens = token_ids
        
        # 绝活体验：立刻去 Radix 树里“白嫖”
        # 比如：给了1000个Token，树里刚好有900个
        self.cached_len, self.cached_kv = global_radix_cache.match_prefix(self.prompt_tokens)
        
        # 剩下的才是真正需要交给GPU从头算的 (比如只剩100个)
        self.unescaped_tokens = self.prompt_tokens[self.cached_len:]
        
        self.generated_tokens = []
        self.status = SequenceStatus.WAITING
        self.max_tokens = max_tokens


# ==========================================
# 绝活二 & 大脑：调度器 Scheduler 与 分块预填充 (Chunked Prefill)
# ==========================================

class SGLangScheduler:
    def __init__(self):
        self.waiting = deque()
        self.running = []
        # SGLang 杀手锏参数：一次前向传播，GPU最多吞的Token数 (Chunk Size)
        # 用来防止 10万字的超长文本一波带走显存
        self.max_chunk_size = 50 

    def schedule(self):
        """调度决策核心阶段"""
        batch_tokens_to_compute = [] # 本次准备发给GPU算的所有Token
        current_batch_size = 0
        scheduled_seqs = []

        # 1. 先看 Running 队列里有没有急需吐下一个字的老客 (Decode 阶段)
        for seq in self.running:
            if current_batch_size < self.max_chunk_size:
                 # Decode 每次只吐1个词，体量很小，优先满足
                 batch_tokens_to_compute.append({"seq": seq, "tokens": [seq.generated_tokens[-1]]})
                 current_batch_size += 1
                 scheduled_seqs.append(seq)

        # 2. 重点！！！处理 Waiting 队列里的新人 (包含 Chunked Prefill 逻辑)
        while self.waiting and current_batch_size < self.max_chunk_size:
            seq = self.waiting[0]
            
            # 还能塞多少Token？
            remaining_quota = self.max_chunk_size - current_batch_size
            
            # 如果新人的词太多（比如还剩80个），一口吃不下，就【切成Chunk】
            if len(seq.unescaped_tokens) > remaining_quota:
                # 只切出 quota 大小的块 (比如50个) 送去算，剩下的留着下一轮算
                chunk_to_compute = seq.unescaped_tokens[:remaining_quota]
                seq.unescaped_tokens = seq.unescaped_tokens[remaining_quota:]
                
                batch_tokens_to_compute.append({"seq": seq, "tokens": chunk_to_compute, "is_chunked": True})
                current_batch_size += remaining_quota
                
                print(f"[调度器] ⚠️ 触发 Chunked Prefill！序列 {seq.seq_id} 太长，本次只切除 {remaining_quota} 个词送算！")
                break # 算力吃满了，本次不再接待新人
            
            else:
                # 算力配额够，一口气把新人剩下的词全算完
                tokens_to_compute = seq.unescaped_tokens
                batch_tokens_to_compute.append({"seq": seq, "tokens": tokens_to_compute, "is_chunked": False})
                current_batch_size += len(tokens_to_compute)
                seq.unescaped_tokens = [] # 清空待算列表
                
                # 新人算完首批，正式转正进入 Decode 阶段
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                self.waiting.popleft()
                scheduled_seqs.append(seq)

        return batch_tokens_to_compute


# ==========================================
# GPU 引擎计算 (模拟) 
# ==========================================

class SGLangEngine:
    def __init__(self, scheduler: SGLangScheduler):
        self.scheduler = scheduler

    def forward_step(self):
        """引擎每次心跳：发一次CUDA Graph给GPU"""
        # 1. 问调度器要活干
        batch_data = self.scheduler.schedule()
        if not batch_data:
            return False # 没活了，停机

        # 2. 【CUDA Graph 加速】(概念演示)
        # SGLang 会在这里避免繁琐的 Python 循环，直接一把梭把建好的 Graph 砸给 GPU
        print("\n--- ⚡ GPU 轰鸣: 开始执行一次 Forward (可能携带 CUDA Graph) ---")
        
        # 3. 模拟 GPU 算完返回的结果
        for item in batch_data:
            seq = item["seq"]
            computed_tokens = item["tokens"]
            
            # 如果这是一个被 Chunked（强行被切开）的 Prefill，
            # 算完这一块后模型没法吐词，只能把新算出来的 KV Cache 暂存起来
            if item.get("is_chunked"):
                print(f"👉 序列 {seq.seq_id} 的 {len(computed_tokens)}个 Chunk 词算完了，更新到了 KV Cache！还没吐新词。")
            else:
                # 如果是正常算完了，模型就会吐出一个新词 (这里随机模拟一个)
                new_token = np.random.randint(1000, 9999)
                seq.generated_tokens.append(new_token)
                print(f"🟢 序列 {seq.seq_id} 吐出了新词: {new_token}")
                
                # 吃饱了就下桌
                if len(seq.generated_tokens) >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.scheduler.running.remove(seq)
                    print(f"✅ 序列 {seq.seq_id} 生成完毕！")

        return True


# ==========================================
# 运行演示
# ==========================================

if __name__ == "__main__":
    print("🌟 启动 mini-SGLang (伪代码演示版) 🌟\n")
    
    # 初始化全局 Radix 树 (假设已经运行了很久，树里缓存了公司的 System Prompt)
    global_radix_cache = RadixCache()
    # 偷偷给树种下点果实 (假设前置缓存了 10 个特定的 Token ID)
    global_radix_cache.root.children[888] = RadixNode([888] * 10) 
    
    scheduler = SGLangScheduler()
    engine = SGLangEngine(scheduler)

    print("--------------------------------------------------")
    print("场景1：Radix Cache 白嫖时刻")
    # 请求1：开局自带大量跟缓存一样的 Prompt (888开头的系统提示词) + 自己的微小提问
    req1_prompt = [888]*10 + [1, 2, 3] 
    seq1 = Sequence(seq_id=1, token_ids=req1_prompt, max_tokens=2)
    # 经过 Sequence 的初始化拦截，你会发现它真正需要算的 unescaped_tokens 只剩 [1, 2, 3]
    print(f"请求1 总长 {len(req1_prompt)}, 命中树缓存 {seq1.cached_len}个，实际要算的只剩 {len(seq1.unescaped_tokens)}个！")
    scheduler.waiting.append(seq1)

    print("--------------------------------------------------")
    print("场景2：超长文本炸塞 & Chunked Prefill 登场")
    # 请求2：一个惊天动地长达 80个 Token 的新请求 (而我们的 GPU 引擎一波最多吞 50个)
    req2_prompt = [999]*80
    seq2 = Sequence(seq_id=2, token_ids=req2_prompt, max_tokens=1)
    print(f"请求2 带来了 {len(req2_prompt)}个无法命中缓存的生词，而引擎最大算力只有 {scheduler.max_chunk_size}！")
    scheduler.waiting.append(seq2)
    print("--------------------------------------------------")

    # 开机，让子弹飞！
    step = 0
    while engine.forward_step():
        step += 1
