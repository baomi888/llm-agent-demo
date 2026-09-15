# llm-agent-demo

第 7 周作业 Demo：**极简 Agent 智能体 —— 工具调用（Tool-Calling）与 ReAct 循环**

不接入 RAG、不做向量检索，专注演示最核心的 Agent 闭环：
**模型自主判断是否调用工具 → 调用工具 → 获取结果 → 整理输出最终答案**。

## Demo 功能：天气实时查询助手（双数据源）

用户输入自然语言问题（中文为主），Agent 自主判断下一步动作：

- **问当前天气** → 调用 `get_current_weather`，返回高德**实时**天气（天气现象、实时气温、湿度、风向风力、发布时间）
- **问未来几天预报** → 调用 `get_weather_forecast`：**1~4 天来自高德**（含风向风力），**5~7 天自动切换 Open-Meteo** 国际气象模型（高德预报上限 4 天）
- **多城市对比**（如"广州和深圳哪里更热"）→ 自主连续调用多次工具后比较
- **追问**（"北京今天怎么样？"→"那明天呢？"）→ **多轮记忆**自动补全省略的城市名
- **中文或拼音输入均可**（`茂名` / `maoming` / `wulumuqi`）：pypinyin + 高德行政区划表动态构建**全国城市拼音索引**（本地缓存，仅首次构建消耗一次配额）
- **询问支持范围** → 调用 `list_supported_cities` 工具
- **闲聊 / 简单数学** → 不调用任何工具，直接回答
- **工具报错**（地名查不到、天数超出 7 天、接口异常/限流）→ 用通俗中文解释原因，绝不编造天气

> **数据说明（全实时、无模拟数据）**：所有天气数据均为实时调用，不做本地模拟兜底；
> 接口失败时工具返回明确错误信息，由 Agent 如实向用户解释。

## 技术栈

- Python 3.10+
- LangChain 0.3（`create_react_agent` + `AgentExecutor` + `ConversationBufferMemory`，经典 ReAct 范式）
- langchain-openai（OpenAI 兼容接口，支持 DeepSeek / 通义 / 智谱等）
- 高德地图 Web 服务 API（地理编码 + 实时天气 + 行政区划表）
- Open-Meteo（5~7 天预报第二数据源，免费无需 key）
- pypinyin（全国城市拼音索引）、python-dotenv（.env 密钥管理）
- 自定义 `ReActJsonArgsOutputParser`：让经典 ReAct Agent 的 `Action Input` 支持多参数 JSON

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai pypinyin python-dotenv
# 运行离线单元测试另需：
pip install pytest
```

### 2. 配置密钥（二选一，系统环境变量优先级更高）

**方式 A：.env 文件（推荐演示用）**——复制 `.env.example` 为 `.env` 并填入真实 key：

```text
LLM_API_KEY=sk-你的key
AMAP_API_KEY=你的高德key
```

> 注意：含真实 key 的 `.env` 不要提交作业或上传仓库！

**方式 B：系统环境变量**：

```powershell
$env:LLM_API_KEY="sk-你的key"        # LLM 的 key（以 DeepSeek 为例）
$env:AMAP_API_KEY="你的高德key"      # 高德 key
```

永久生效（Windows）：

```powershell
[Environment]::SetEnvironmentVariable("LLM_API_KEY","sk-你的key","User")
[Environment]::SetEnvironmentVariable("AMAP_API_KEY","你的高德key","User")
```

**高德 Key 申请**（免费）：[高德开放平台控制台](https://console.amap.com/dev/key/app) → 应用管理 → 创建应用 → 添加 Key，**服务平台必须选「Web服务」**（选错会报 `INVALID_USER_KEY`）。

可选环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容接口地址 |
| `WEATHER_HTTP_TIMEOUT` | `10` | 天气/地理编码接口超时秒数 |

### 3. 运行

**交互模式（默认）—— 多轮对话 + 记忆追问：**

```bash
python main.py
```

```text
你 > 北京今天天气怎么样？
Agent > 北京今天天气晴朗，当前实时气温 22℃，湿度 72%……（数据来源：高德实时天气）
你 > 那明天呢？
Agent > 北京明天（9月15日 周二）天气晴，气温 18~28℃……（自动补全"北京"）
```

输入 `q` / `quit` / `exit` / `退出` 或 `Ctrl+C` 结束对话。

**固定测试用例模式（汇报截图用）：**

```bash
python main.py --test
```

**离线单元测试（不访问网络、不消耗 API 配额）：**

```bash
python -m pytest test_main.py -v
```

## 内置测试用例

覆盖「实时调用成功 / 多轮工具调用 / 调用工具失败 / 不调用工具」等路径：

| # | 问题 | 预期路径 |
|---|---|---|
| 1 | 北京今天天气怎么样？ | 调用 get_current_weather，返回高德实时天气 |
| 2 | 上海未来3天的天气预报给我看看 | 调用 get_weather_forecast(days=3)，高德逐日预报 |
| 3 | 广州和深圳今天哪里更热？ | 连续两次调用 get_current_weather，对比实时气温 |
| 4 | 浙江义乌今天天气怎么样？ | 县级市动态解析，验证全国城市实时覆盖 |
| 5 | 你都支持查询哪些城市的天气？ | 调用 list_supported_cities，说明全国范围 |
| 6 | 你好，你是谁？ | 不调用工具，直接回答 |
| 7 | 1+1等于多少？ | 不调用工具，直接回答 |
| 8 | 火星上的天气怎么样？ | 地理编码查无此地，Agent 解释只支持中国城市 |
| 9 | 帮我预测一下北京30天后的天气 | days=30 超出 1~7，Agent 解释最多支持未来 7 天 |

运行时可在 `verbose` 日志中观察完整闭环：
`Thought(思考) → Action(行动) → Action Input(参数) → Observation(观察) → Final Answer(回答)`。

## ReAct 闭环与代码结构对照

| ReAct 环节 | 代码位置 |
|---|---|
| 工具定义（Action 的能力来源） | `@tool get_current_weather / get_weather_forecast / list_supported_cities` |
| Thought / Action 输出格式约束 | `REACT_PROMPT`（中文 ReAct 模板，含多轮追问规则） |
| Action Input JSON 多参数解析 | `ReActJsonArgsOutputParser` |
| 思考→行动→观察循环驱动 | `create_react_agent` + `AgentExecutor(verbose=True)` |
| 多轮记忆（追问补全上下文） | `ConversationBufferMemory(memory_key="chat_history")` |
| 城市名 → adcode + 经纬度（参数落地） | `resolve_city()`（高德地理编码 + 拼音索引 + 误匹配校验） |
| 全国城市拼音输入 | `_load_pinyin_index()`（高德行政区划表 + pypinyin，磁盘缓存） |
| 实时天气 / 1~4 天预报 | `_fetch_live_weather()` / `_fetch_forecast()`（高德） |
| 5~7 天预报 | `_fetch_open_meteo_daily()`（Open-Meteo，经纬度查询） |

## 文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | Agent 主程序：3 个天气工具 + ReAct 组装 + 记忆 + 交互/测试双模式 |
| `test_main.py` | pytest 离线单元测试：全部 mock HTTP，18 条用例覆盖数据层/工具层/解析器/提示词 |
| `.env.example` | 密钥配置模板（复制为 `.env` 使用） |
| `.amap_districts.json` | 运行时自动生成的全国行政区划拼音索引缓存（可删除，会自动重建） |
| `README.md` | 本说明 |

## 学习参考

- 论文：ReAct: Synergizing Reasoning and Acting in Language Models
- 文档：LangChain 官方 Agent 示例文档、Custom Tools 自定义工具章节、`create_react_agent` API
- 数据源：[高德开放平台 Web 服务 API](https://lbs.amap.com/api/webservice/summary)、[Open-Meteo](https://open-meteo.com/)（均为个人免费额度）
