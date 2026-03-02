# 全面解析：A2A (Agent-to-Agent) 通信协议

你提到的 `A2A (Agent-to-Agent) Protocol` 是当前 AI 圈非常前沿的一个标准！

它最初是由 Google 在 2025 年提出（现已捐赠给 Linux 基金会）的开源协议规范。它的核心野心是：**打破目前各个 AI 框架（比如 LangChain, AutoGen, MetaGPT 等）各自为战的系统孤岛，让全世界不同厂商、不同语言编写的 AI 智能体（Agent）能够像人类用微信聊天一样，顺畅地跨系统交流、派发任务和共享文件。**

如果说 MCP（Model Context Protocol）是解决“大模型如何调用外部工具（API/数据库）”的协议，那么 **A2A 则是解决“一个智能体如何呼叫另一个智能体”的通信宪法。**

---

## 一、 A2A 的核心通信机制：它是怎么传输的？

A2A 并没有自己发明一套底层的魔幻网络协议，它极其务实：

- **传输层**：全面基于安全的 **HTTPS**。
- **数据格式**：全面采用轻量、无状态的 **JSON-RPC 2.0** 规范。

这就是你关心的第一个问题：**调用参数是怎么传递的？**
答案是：**所有的接口调用、参数传递，全部被封装成一个标准的 JSON-RPC 报文，通过 HTTP POST 请求发送给对方 Agent 的接听端口（Endpoint）。**

---

## 二、 核心接口拆解：参数是如何传递的？

在 A2A 协议中，无论你调用什么接口，发出去的“网络包裹”永远长成这个标准的 JSON 格式：

```json
{
  "jsonrpc": "2.0",
  "method": "这里写调用的接口名称",
  "params": {
    // 这里放该接口需要的具体的复杂参数对象
  },
  "id": "请求的唯一流水号"
}
```

我们以 A2A 最核心、最常用的接口 **`message/send` (发送消息/指派任务)** 为例，看看参数到底长什么样：

### 核心接口实例：`message/send`

假设 **Agent A（客户端）** 想让 **Agent B（服务端/专家）** 帮忙分析一份财务数据。Agent A 会向 Agent B 发送如下的大 JSON 包裹：

```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg_987654321",
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "请帮我提取这份财报里的 Q3 净利润。"
        },
        {
          "kind": "file",
          "mimeType": "application/pdf",
          "url": "https://storage.example.com/finance_report_Q3.pdf"
        }
      ]
    },
    "metadata": {
      "priority": "high",
      "taskId": "task_112233"
    }
  },
  "id": "req_001"
}
```

### 参数字典详解：

1. **`method`**: 这是路由地址。比如 `message/send` 就是告诉对方，这是一条聊天指令或任务指令。其他常见的 Method 还有：
   - `task/status`：查询之前交办的任务进度到了哪里。
   - `task/cancel`：取消一个正在执行的巨型任务。
2. **`params` (重头戏)**: 所有的动态业务数据都在这里！
   - **`role`**: 标明是谁在说话（通常是 `user` 或 `agent`）。
   - **`parts`**: 极度灵活的数组！A2A 协议不仅支持传纯文本 (`text`)，还原生支持多模态！如果你想传文件，只要在 `parts` 里把 `kind` 设为 `file`，并贴上文件的 `url` 即可。Agent B 收到后自己去下载。
   - **`metadata`**: 存放上下文数据（比如这个任务所属的流程追踪 ID，用于多轮对话对齐）。
3. **`id`**: 这是异步通信的灵魂。因为 Agent 思考可能要花 5 分钟！Agent B 收到后，会立刻回绝一个带有 `id: "req_001"` 的回执，告诉你“我收到了”。等 5 分钟后它算完了，它会拿着这个 `id` 再回调告诉你最终答案。

---

## 三、 A2A 协议的另外两个神级设计

为什么它能一统江湖？除了标准的 JSON-RPC 传参数外，它还解决了一长一短两个痛点：

### 1. Agent Discovery：智能体怎么互相发现对方的接口？（Agent Card）

在人与人的世界，我们递名片。在 A2A 世界，这叫 **`Agent Card`**。
这是一个可以通过统一 URL 访问的 JSON 元数据文件。如果 Agent A 想联系一个新的 Agent C，它可以先请求获取 Agent C 的 Agent Card，里面会白纸黑字写着：

- **`contact_endpoints`**：我支持哪些 HTTPS 地址通信。
- **`supported_methods`**：我支持调用哪些 RPC 接口（比如我只支持聊天，不支持传文件）。
  这样，不同大厂捏出来的 Agent，只要一见面交换一下 Agent Card，立马就知道该怎么用参数调对方了。

### 2. 身份认证 (Authentication)

既然是互相调参数，怎么防止恶意黑客 Agent 套话？
A2A 复用了极其成熟的 **OpenAPI 安全规范**。它支持在 HTTP Header 里塞入：

- **API Keys**：最简单的 Bearer Token。
- **OAuth 2.0 / OpenID**：通过第三方鉴权中心发放临时的短效 Token 进行跨域安全认证。

## 总结你的应用场景

如果你在写一个微服务（比如基于 Go 写的 Higress 网关或者云原生调度器），你想让你系统里的大模型/智能体遵循 A2A 协议与其他系统交互：
**你只需要写一套标准的 HTTP + JSON-RPC 2.0 封包/解包器。**
不管底层模型是 DeepSeek 还是 GPT-4，只要你在网关出口，把大模型的意图翻译成带有 `method` 和 `params` 的 JSON，你就在使用标准的 A2A 协议和全世界的智能体对话了！
