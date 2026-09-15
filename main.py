# -*- coding: utf-8 -*-
"""
第七周作业 Demo：极简 Agent —— 工具调用（Tool-Calling）与 ReAct 循环
=====================================================================
Demo 主题：天气实时查询助手（高德地图数据源）

闭环演示（不涉及 RAG / 向量检索）：
    用户自然语言提问
        -> 【Thought 思考】模型自主判断：要不要调工具？调哪个？
        -> 【Action 行动】调用 get_current_weather / get_weather_forecast / list_supported_cities
        -> 【Observation 观察】拿到工具返回（实时天气结果 / 错误信息）
        -> 【Final Answer 回答】整理成通顺中文；闲聊、简单数学则不调工具直接回答

数据说明（全实时，无模拟数据，双数据源）：
    1~4 天   ：高德地图开放平台 Web 服务 API（https://lbs.amap.com）
      - 地理编码接口：把任意中国省/市/区县名动态解析为行政区划代码 adcode（覆盖全国）
      - 天气查询接口：按 adcode 返回实时天气（实况）与未来 4 天天气预报
    5~7 天   ：Open-Meteo 国际气象模型（https://open-meteo.com，免费无需 key），
               用地理编码返回的经纬度查询；高德预报上限为 4 天，故 5~7 天走第二数据源。
    按要求：所有城市的数据均为实时调用，不做本地模拟兜底；
    接口失败时工具返回明确错误信息，由 Agent 如实向用户解释。

其他能力：
    - 全国城市拼音输入：pypinyin + 高德行政区划表动态构建拼音索引（本地缓存，
      仅首次调用消耗一次高德配额），任何省市县都能用拼音查（如 maoming、wulumuqi）
    - 多轮记忆：交互模式下记住本次会话的问答历史，支持“那明天呢？”这类追问
    - 离线单元测试：test_main.py 用 pytest + mock 全部 HTTP，不消耗任何 API 配额

依赖安装：
    pip install langchain langchain-openai pypinyin python-dotenv
    （运行离线单元测试另需：pip install pytest）

密钥配置方式（二选一，环境变量优先级更高）：
    A) 系统环境变量（Windows PowerShell 示例）：
       $env:LLM_API_KEY="sk-你的key"          # LLM 的 key（以 DeepSeek 为例）
       $env:AMAP_API_KEY="你的高德key"        # 高德 key：console.amap.com 注册 ->
                                              # 应用管理 -> 创建应用 -> 添加 Key，
                                              # 服务平台必须选「Web服务」
    B) .env 文件：复制 .env.example 为 .env 并填入真实 key（python-dotenv 自动读取，
       注意不要把含真实 key 的 .env 提交作业或上传仓库）
可选环境变量：
    LLM_MODEL             默认 deepseek-chat
    LLM_BASE_URL          默认 https://api.deepseek.com/v1
    WEATHER_HTTP_TIMEOUT  默认 10（天气/地理编码接口超时秒数）
切换其他 LLM 厂商示例：
    通义千问：LLM_MODEL=qwen-plus  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    智谱GLM ：LLM_MODEL=glm-4-flash LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import warnings
from typing_extensions import override  # 兼容 Python 3.10（3.12 后可直接用 typing.override）

# ============================================================
# ① 集中配置区：模型名 / API Key / 接口地址全部从环境变量读取
# ============================================================
# 可选：读取项目根目录的 .env 文件（python-dotenv，未安装则跳过）。
# 注意 load_dotenv 默认不覆盖已存在的系统环境变量，因此系统变量优先级更高
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")          # 模型名
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")  # OpenAI 兼容接口地址
# 兼容常见变量名：LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
# 高德开放平台 key（「Web服务」类型），天气与地理编码都靠它
AMAP_API_KEY = os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY") or ""
WEATHER_HTTP_TIMEOUT = float(os.getenv("WEATHER_HTTP_TIMEOUT", "10"))  # 高德接口超时秒数

from langchain_core.tools import tool          # @tool 装饰器：把普通函数变成 Agent 可调用的工具
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction   # ReAct 解析产物：一次“行动”
from langchain_openai import ChatOpenAI         # OpenAI 兼容协议的聊天模型
from langchain.agents import create_react_agent, AgentExecutor  # ReAct Agent 组装器 + 执行器
# 经典 ReAct 默认解析器把 Action Input 一律当“单字符串”，不支持多参数 JSON，
# 因此下面继承它做一层 JSON 解析增强（兼容 city/days 多参数工具）
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ConversationBufferMemory  # 多轮记忆：保存本次会话问答历史

# ConversationBufferMemory 在 langchain 0.3 中标记为“推荐迁移 LangGraph”，
# 但作为课程演示的经典做法仍完全可用；这里静默该弃用提醒，保持输出干净
warnings.filterwarnings("ignore", message=".*ConversationBufferMemory.*")


# ============================================================
# ② 高德实时天气数据层（与 Agent 解耦的纯 Python 部分，仅用标准库 urllib）
# ============================================================
import time
import urllib.error
import urllib.parse
import urllib.request

_AMAP_GEO_URL = "https://restapi.amap.com/v3/geocode/geo"          # 地理编码：地名 -> adcode
_AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"  # 天气：实况 + 预报
_AMAP_DISTRICT_URL = "https://restapi.amap.com/v3/config/district"     # 行政区划：全国省市县树
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"             # 5~7 天预报第二数据源

# 拼音/英文别名 -> 中文名（高德地理编码对拼音支持不稳定，先归一化再查询）
CITY_ALIASES = {
    "beijing": "北京", "tianjin": "天津", "xian": "西安", "harbin": "哈尔滨",
    "shanghai": "上海", "hangzhou": "杭州", "nanjing": "南京", "suzhou": "苏州",
    "wuhan": "武汉", "chengdu": "成都", "chongqing": "重庆", "guangzhou": "广州",
    "shenzhen": "深圳", "haikou": "海口", "kunming": "昆明", "lhasa": "拉萨",
    "maoming": "茂名", "shaoguan": "韶关", "qingyuan": "清远", "zhuhai": "珠海",
    "xiamen": "厦门", "fuzhou": "福州", "jinan": "济南", "qingdao": "青岛",
    "zhengzhou": "郑州", "changsha": "长沙", "shijiazhuang": "石家庄", "guiyang": "贵阳",
    "nanning": "南宁", "urumqi": "乌鲁木齐", "taiyuan": "太原", "xining": "西宁",
    "lanzhou": "兰州", "yinchuan": "银川", "hohhot": "呼和浩特", "changchun": "长春",
    "shenyang": "沈阳", "dalian": "大连", "taipei": "台北", "hongkong": "香港",
    "macau": "澳门", "macao": "澳门",
}

# 高德偶发的可重试错误码：QPS 限流 / 服务端引擎瞬时错误
_TRANSIENT_INFO_CODES = {"10021", "10022", "30001"}

# 进程内缓存：城市名 -> (标准名, adcode)。行政区划代码长期不变，可永久缓存，
# 同一城市当天多次提问不会重复消耗高德地理编码配额
_ADCODE_CACHE: dict = {}

_WEEKDAY = "周一 周二 周三 周四 周五 周六 周日".split()

# Open-Meteo 返回 WMO 标准天气码 -> 中文描述（供 5~7 天预报使用）
_WMO_CN = {
    0: "晴", 1: "基本晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "浓毛毛雨", 56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "阵雨", 82: "强阵雨", 85: "阵雪", 86: "阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹",
}

# 行政区划名称的常见后缀（拼音索引与展示时统一去掉）
_CITY_SUFFIX_RE = re.compile(r"(市|地区|自治州|自治县|盟|林区|新区)$")


def _http_get_json(base_url: str, params: dict, retries: int = 2, amap_status: bool = True) -> dict:
    """用标准库发 GET 请求并解析 JSON；高德接口自动附带 key。

    健壮性：QPS 限流 / 服务端瞬时错误 / 网络抖动会自动重试（间隔递增），
    最终失败时抛出带高德错误码的异常，由工具层转为如实错误信息。
    amap_status=False 时用于 Open-Meteo（无 status 字段，HTTP 200 即成功）。
    """
    if amap_status:
        params = dict(params, key=AMAP_API_KEY)
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-agent-demo/1.0"})
            with urllib.request.urlopen(req, timeout=WEATHER_HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not amap_status:
                return data                     # Open-Meteo：无 status 字段
            if str(data.get("status")) == "1":
                return data
            info = f"{data.get('info')}（infocode={data.get('infocode')}）"
            # QPS 限流 / 引擎瞬时错误：短暂等待后重试
            if str(data.get("infocode")) in _TRANSIENT_INFO_CODES and attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise RuntimeError(info)
        except (urllib.error.URLError, TimeoutError) as exc:  # 网络抖动/超时也重试
            if attempt < retries:
                time.sleep(0.6 * (attempt + 1))
                continue
            raise RuntimeError(f"网络请求失败（{type(exc).__name__}: {exc}）") from exc
    raise RuntimeError("高德接口请求失败：重试次数已用尽")  # 理论上到不了这里


def resolve_city(name: str):
    """【对应 ReAct 的“参数落地”环节】把城市名解析为 (标准名, adcode, 经纬度)；查不到返回 None。

    任意中国省/市/区县都能解析（如 杭州/义乌/喀什/朝阳区），实现“全国城市都实时查询”。
    中文走高德地理编码；纯字母输入（拼音）先查本地拼音索引（行政区划表 + pypinyin 动态构建）。
    经纬度供 Open-Meteo 查询 5~7 天预报使用。
    """
    raw = str(name).strip()
    query = CITY_ALIASES.get(raw.lower(), raw)   # 拼音先走别名表归一化为中文
    if query in _ADCODE_CACHE:
        return _ADCODE_CACHE[query]

    # 纯字母输入（别名表未命中）：查本地拼音索引，不消耗高德地理编码配额
    if re.fullmatch(r"[a-zA-Z]+", query):
        hit = _load_pinyin_index().get(query.lower())
        if hit is None:
            return None                          # 拼音也查不到（如 huoxing）
        _rank, std_name, adcode, location = hit
        if not location:
            # 区划表 base 模式可能不含城市级中心点：地理编码兜底补齐经纬度（5~7 天预报需要）
            try:
                g = (_http_get_json(_AMAP_GEO_URL, {"address": std_name}).get("geocodes") or [{}])[0]
                location = str(g.get("location") or "")
            except Exception:                    # 兜底失败不阻断 1~4 天查询
                location = ""
        std = _CITY_SUFFIX_RE.sub("", std_name) or std_name
        result = (std, str(adcode), location)
        _ADCODE_CACHE[query] = result
        _ADCODE_CACHE[std] = result              # 中文名也入缓存，两种输入共用
        return result

    # 中文输入：走高德地理编码
    data = _http_get_json(_AMAP_GEO_URL, {"address": query})
    geocodes = data.get("geocodes") or []
    if not geocodes:
        return None                              # 高德查无此地（如“霍格沃茨”）
    g = geocodes[0]
    adcode = str(g.get("adcode") or "")
    if not adcode.isdigit():
        return None
    # 名称包含校验：防止短词模糊命中不相干地名（如“火星”误匹配到某地“火星村”）
    prov = str(g.get("province") or "")
    city = g.get("city")
    city = str(city) if isinstance(city, str) else ""
    dist = str(g.get("district") or "")
    if query not in (prov + city + dist):
        return None                              # 匹配结果与查询词不符，视为查无此地
    # 标准名：优先取最精确的级别（区县 > 城市 > 省），直辖市的 city 为空列表则逐级回退
    std = dist or city or prov or query
    std = _CITY_SUFFIX_RE.sub("", std) or str(std)  # 去掉“市/地区”等后缀，展示更简洁
    result = (std, adcode, str(g.get("location") or ""))
    _ADCODE_CACHE[query] = result
    return result


def _fetch_live_weather(adcode: str) -> dict:
    """调用高德「实况」接口（extensions=base），返回当前实时天气字典。"""
    data = _http_get_json(_AMAP_WEATHER_URL, {"city": adcode, "extensions": "base"})
    lives = data.get("lives") or []
    if not lives:
        raise RuntimeError("实况接口未返回数据")
    return lives[0]


def _fetch_forecast(adcode: str) -> list:
    """调用高德「天气预报」接口（extensions=all），返回今天起共 4 天的逐日预报列表。"""
    data = _http_get_json(_AMAP_WEATHER_URL, {"city": adcode, "extensions": "all"})
    forecasts = data.get("forecasts") or []
    if not forecasts or not forecasts[0].get("casts"):
        raise RuntimeError("预报接口未返回数据")
    return forecasts[0]["casts"]                 # 每项含 date/dayweather/nightweather/daytemp/...


# 进程内拼音索引：pinyin -> (级别序, 标准名, adcode, 经纬度)。None 表示尚未构建
_PINYIN_INDEX: dict | None = None
# 行政区划表磁盘缓存（约几千条，仅首次构建时请求一次高德；删除该文件可强制刷新）
_DISTRICT_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".amap_districts.json")


def _flatten_districts(nodes: list, out: list) -> None:
    """递归展平高德行政区划树（省 -> 市 -> 区县），收集 (名称, adcode, 级别, 经纬度)。"""
    for d in nodes or []:
        name, adcode = d.get("name"), str(d.get("adcode") or "")
        if name and adcode.isdigit():
            out.append((name, adcode, str(d.get("level") or ""), str(d.get("location") or "")))
        _flatten_districts(d.get("districts"), out)


def _load_pinyin_index() -> dict:
    """构建（并缓存）“全国行政区划拼音索引”：pinyin -> (级别序, 名称, adcode, 经纬度)。

    数据来源：高德行政区划接口（全国省/市/区县一次性拉取，进程内 + 磁盘双重缓存）；
    拼音由 pypinyin 动态生成，因此任何城市/区县都支持拼音输入，无需手工维护别名表。
    同音重名时择优：级别 higher 优先（市 > 区县），保证 yichun 等优先命中地级市。
    """
    global _PINYIN_INDEX
    if _PINYIN_INDEX is not None:
        return _PINYIN_INDEX

    entries = None
    if os.path.exists(_DISTRICT_CACHE_FILE):     # 1) 磁盘缓存
        try:
            with open(_DISTRICT_CACHE_FILE, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            entries = None
    if not entries:                              # 2) 首次：请求高德行政区划接口
        data = _http_get_json(_AMAP_DISTRICT_URL, {"keywords": "中国", "subdistrict": 3})
        root = (data.get("districts") or [{}])[0].get("districts") or []
        flat: list = []
        _flatten_districts(root, flat)
        entries = [list(t) for t in flat]
        try:                                     # 写磁盘缓存失败不影响运行
            with open(_DISTRICT_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False)
        except OSError:
            pass

    from pypinyin import lazy_pinyin             # 惰性导入，中文查询路径不依赖它
    level_rank = {"province": 0, "city": 1, "district": 2}
    index: dict = {}
    for name, adcode, level, location in entries:
        rank = level_rank.get(level, 3)
        for key in {_CITY_SUFFIX_RE.sub("", name) or name, name}:  # “茂名市”/“茂名”都入索引
            pinyin = "".join(lazy_pinyin(key)).lower()
            old = index.get(pinyin)
            if old is None or rank < old[0]:     # 同音重名：级别更高者胜出
                index[pinyin] = (rank, name, adcode, location)
    _PINYIN_INDEX = index
    return _PINYIN_INDEX


def _fetch_open_meteo_daily(location: str, days: int) -> list:
    """调用 Open-Meteo（免费无 key）按经纬度查询未来 days 天逐日预报。

    高德预报上限为 4 天，5~7 天由本函数提供。返回 [{date, day, max, min}, ...]。
    """
    try:
        lon, lat = str(location).split(",")      # 高德经纬度格式："经度,纬度"
    except ValueError:
        raise RuntimeError("缺少可用的城市经纬度") from None
    data = _http_get_json(_OPEN_METEO_URL, {
        "latitude": lat.strip(), "longitude": lon.strip(),
        "daily": "weather_code,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Shanghai", "forecast_days": days,
    }, amap_status=False)
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    if not times:
        raise RuntimeError("Open-Meteo 未返回预报数据")
    return [{"date": times[i],
             "day": _WMO_CN.get(codes[i], "未知"),
             "max": tmax[i], "min": tmin[i]}
            for i in range(min(days, len(times)))]


def _fmt_cast_date(date_str: str) -> str:
    """把高德预报日期（YYYY-MM-DD）格式化为“9月16日 周三”。"""
    y, m, d = map(int, date_str.split("-"))
    return f"{m}月{d}日 {_WEEKDAY[dt.date(y, m, d).weekday()]}"


# ============================================================
# ③ 自定义工具（LangChain @tool 装饰器）
#    函数名 -> Action 名；docstring -> 给模型看的工具说明（决定何时调用、调哪个）
# ============================================================

@tool
def get_current_weather(city: str) -> str:
    """查询任意中国城市/区县“当前”的实时天气（天气现象、实时气温、湿度、风向风力、发布时间）。

    适用场景：用户问“某城市/区县今天天气怎么样 / 现在多少度 / 冷不冷 / 下不下雨”等实时天气问题。
    支持全国所有省/市/区县（如 北京、杭州、义乌、喀什、上海朝阳区）。
    多城市对比时可对每个城市分别调用一次本工具。
    若返回“未找到”错误，说明高德地理编码查无此地，应如实告知用户，绝不能编造天气。

    Args:
        city: 城市/区县中文名或常见拼音，例如 "北京"、"杭州市"、"义乌"、"beijing"。

    Returns:
        实时天气文本（含数据发布时间）；地名不存在或接口异常时返回明确的中文错误说明。
    """
    # —— 对应 ReAct 的 Action 环节：真正执行工具 ——
    try:
        resolved = resolve_city(city)
    except Exception as exc:  # 网络异常 / key 无效等，如实报告，不用模拟数据兜底
        return f"错误：高德地理编码接口请求失败（{type(exc).__name__}: {exc}）。"
    if resolved is None:
        return (f"错误：在高德地理编码中未找到“{city}”，无法查询天气。"
                f"本助手仅支持真实存在的中国省/市/区县，请确认地名后重试。")
    std, adcode, _location = resolved
    try:
        live = _fetch_live_weather(adcode)
    except Exception as exc:
        return f"错误：高德实时天气接口请求失败（{type(exc).__name__}: {exc}）。"
    return (f"{std}当前{live.get('weather', '未知')}，"
            f"实时气温 {live.get('temperature', '?')}℃，"
            f"湿度 {live.get('humidity', '?')}%，"
            f"{live.get('winddirection', '?')}风 {live.get('windpower', '?')}级。"
            f"（高德实时数据，发布于 {live.get('reporttime', '?')}）")


@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """查询任意中国城市/区县“未来若干天”的天气预报（逐天给出天气、气温范围、风向风力）。

    适用场景：用户问“明天/后天/未来N天天气怎么样、会不会下雨、要不要带伞”等。
    中文相对时间换算：明天=1天、后天=2天、大后天=3天。
    数据源约束：days 必须是 1~7 的整数。1~4 天来自高德（含风向风力）；
    5~7 天高德不支持，自动改用 Open-Meteo 国际气象模型（仅气温与天气现象）。
    超出范围会返回错误，此时应向用户说明可预报范围，不能编造更久远的天气。

    Args:
        city: 城市/区县中文名或拼音，例如 "杭州"、"义乌"、"maoming"。
        days: 预报天数，取 1~7 的整数（今天起算的未来 N 天），默认 3。

    Returns:
        逐日预报文本（标注数据来源）；地名不存在或天数非法时返回明确的中文错误说明。
    """
    if not isinstance(days, int) or isinstance(days, bool) or days < 1 or days > 7:
        return (f"错误：预报天数 days 必须是 1~7 的整数，收到的是 {days!r}；"
                f"高德（1~4 天）+ Open-Meteo（5~7 天）双数据源最多支持未来 7 天预报。")
    try:
        resolved = resolve_city(city)
    except Exception as exc:
        return f"错误：高德地理编码接口请求失败（{type(exc).__name__}: {exc}）。"
    if resolved is None:
        return (f"错误：在高德地理编码中未找到“{city}”，无法查询天气。"
                f"本助手仅支持真实存在的中国省/市/区县，请确认地名后重试。")
    std, adcode, location = resolved

    lines = [f"{std}未来 {days} 天天气预报："]
    try:
        if days <= 4:
            # —— 1~4 天：高德预报（含白天/夜间天气、风向风力）——
            for cast in _fetch_forecast(adcode)[:days]:
                lines.append(
                    f"{_fmt_cast_date(cast.get('date', ''))}："
                    f"白天{cast.get('dayweather', '?')}，夜间{cast.get('nightweather', '?')}，"
                    f"{cast.get('nighttemp', '?')}~{cast.get('daytemp', '?')}℃，"
                    f"{cast.get('daywind', '?')}风 {cast.get('daypower', '?')}级。"
                )
            lines.append("（高德天气预报数据，每日多次更新）")
        else:
            # —— 5~7 天：高德不支持，走 Open-Meteo 第二数据源 ——
            for d in _fetch_open_meteo_daily(location, days):
                lines.append(
                    f"{_fmt_cast_date(d['date'])}："
                    f"{d['day']}，{d['min']}~{d['max']}℃。"
                )
            lines.append("（5~7 天预报来自 Open-Meteo 国际气象模型，含天气与气温；"
                         "高德数据仅覆盖未来 4 天）")
    except Exception as exc:
        return f"错误：天气预报接口请求失败（{type(exc).__name__}: {exc}）。"
    return "\n".join(lines)


@tool
def list_supported_cities() -> str:
    """查询本助手支持的城市范围。当用户询问“支持哪些城市 / 能查哪里 / 你会什么”时调用。不需要任何参数。"""
    return ("本助手基于高德地图实时数据，支持全国所有省/市/区县级城市的天气查询，"
            "例如北京、上海、广州、杭州、义乌、喀什、茂名、三沙、香港、澳门等均可直接查询；"
            "所有城市都支持拼音输入（如 beijing、maoming、wulumuqi），中文与拼音均可。")


# Agent 可路由的工具清单：模型会在这三个工具之间自主选择（或选择都不用）
TOOLS = [get_current_weather, get_weather_forecast, list_supported_cities]


# ============================================================
# ④ ReAct 提示词（中文版经典 ReAct 模板）
#    规定 思考 Thought -> 行动 Action -> 观察 Observation -> 最终回答 的格式
#    注意：占位符 {tools} / {tool_names} / {input} / {agent_scratchpad} 不可改名
# ============================================================
REACT_PROMPT = PromptTemplate.from_template(
    """你是一个“天气实时查询助手”，用中文回答用户问题。你可以查询全国任意省/市/区县的实时天气与未来最多 7 天的天气预报（1~4 天来自高德地图，5~7 天来自 Open-Meteo）。

你可以使用以下工具：
{tools}

请严格按照下面的 ReAct 格式进行推理（每一轮只能输出一个动作）：

Question: 用户的问题
Thought: 你时刻思考下一步该做什么
Action: 要执行的动作名称，只能是 [{tool_names}] 之一；如果不需要工具，则不要出现 Action
Action Input: 动作输入，必须是合法 JSON（若工具无参数则填空对象），例如：{{"city": "北京", "days": 3}}
Observation: 工具返回的结果
（Thought / Action / Action Input / Observation 可根据需要重复多轮，例如对比多个城市可连续调用多次工具）
Thought: 我已经掌握足够信息，知道最终答案了
Final Answer: 给用户的最终中文回答

行为准则：
1. 问“今天/现在/实时”的天气：调用 get_current_weather；问“明天/后天/未来N天/预报”：调用 get_weather_forecast，明天 days=1、后天 days=2，最多 days=7（1~4 天来自高德，5~7 天自动走 Open-Meteo，无需关心细节）。
2. 多城市对比（如“广州和深圳哪个热”）：分别调用工具拿到各城市数据后再比较回答。
3. 询问支持城市/能力范围：调用 list_supported_cities。
4. 闲聊、身份问题、简单常识与数学口算等无需工具的问题：直接给出 Final Answer，禁止调用任何工具。
5. 工具返回错误时（地名查不到、天数超出 1~7、接口异常等）：把错误原因用通俗的中文解释给用户，并给出可行建议，绝不编造天气数据。
6. 回答天气时可给出穿衣/带伞等温馨提示；实时数据请注明是高德实时数据。
7. 【强制】只要用户的问题里出现任何地点（包括但不限于城市、区县、国家、山脉、星球如火星月球、虚构地点）并询问天气/气温/冷热/预报，你必须先调用对应天气工具、以工具返回结果为准；严禁仅凭自身知识判断“不支持/查不了”而跳过工具。即使你确信该地点不支持，也必须先调用一次工具拿到错误信息，再向用户解释。
8. Action Input 必须直接输出 JSON 对象本身，严禁使用 ``` 代码块或任何额外文字包裹；
   无参工具也要输出空对象 {{}}。
9. 用户带上下文追问时（如先问北京天气、再问“那明天呢”）：结合对话记录补全省略的城市名后再调用工具。

（可选）之前的对话记录（多轮追问时结合上下文，首次对话此处为空）：
{chat_history}

现在开始！

Question: {input}
Thought:{agent_scratchpad}"""
)


# ============================================================
# ④-补 自定义 ReAct 输出解析器：让 Action Input 支持“多参数 JSON”
# ------------------------------------------------------------
# 父类 ReActSingleInputOutputParser 的定位是“单输入工具”（如搜索词），
# 它解析出的 tool_input 永远是字符串；get_weather_forecast 需要 city/days 两个具名参数，
# 故在父类解析完成后，把形如 {"city": "北京", "days": 3} 的字符串
# 再 json.loads 成 dict —— 这正对应 ReAct 中“把模型的行动指令落地为可执行调用”。
# ============================================================
class ReActJsonArgsOutputParser(ReActSingleInputOutputParser):
    """支持 JSON 多参数 Action Input 的 ReAct 解析器。"""

    _FENCE_RE = re.compile(r"^\s*```(?:json)?|```\s*$", re.IGNORECASE)

    @override
    def parse(self, text: str):
        result = super().parse(text)   # 先走经典解析：得到 AgentAction 或 AgentFinish

        # AgentFinish（Final Answer）无需处理；只给“行动”补 JSON 解析
        if not isinstance(result, AgentAction) or not isinstance(result.tool_input, str):
            return result

        raw = result.tool_input.strip()
        raw = self._FENCE_RE.sub("", raw).strip()  # 容错：剥掉模型可能输出的 ```json 围栏

        # 只有长得像 JSON 对象时才解析；普通字符串输入则原样保留（向后兼容单参工具）
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return result  # JSON 非法时保留原样，让后续校验/错误回灌机制处理
            if isinstance(parsed, dict):
                return AgentAction(
                    tool=result.tool,
                    tool_input=parsed,   # dict 形式 -> 按具名参数绑定到工具函数
                    log=result.log,
                )
        return result


def build_agent_executor(enable_memory: bool = True) -> AgentExecutor:
    """组装 ReAct Agent：LLM + 工具 + ReAct 提示词 -> Agent 执行器（开启 verbose）。

    enable_memory=True 时附带对话记忆（交互模式支持“那明天呢？”这类追问）；
    False 用于离线单测/逐条独立用例，避免上下文互相污染。
    """
    # —— 模型：OpenAI 兼容接口，temperature=0 让工具选择更稳定 ——
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
    )

    # 多轮记忆：以文本形式把历史问答注入提示词的 {chat_history} 占位符
    memory = ConversationBufferMemory(memory_key="chat_history") if enable_memory else None
    # 无记忆场景（--test 每条用例独立执行）：给 chat_history 预填空串，
    # 避免提示词格式化时缺变量报错；有记忆时由 memory 注入并覆盖该 partial 值
    prompt = REACT_PROMPT if memory is not None else REACT_PROMPT.partial(chat_history="")

    # create_react_agent 对应 ReAct 论文的推理-行动交错范式：
    #   模型每轮输出 Thought/Action，框架执行工具后把 Observation 追加回上下文
    # output_parser：注入自定义解析器，使 Action Input 的 JSON 能绑定为多参数
    react_agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt,
        output_parser=ReActJsonArgsOutputParser(),
    )

    # AgentExecutor 负责驱动“思考→行动→观察”循环；
    # verbose=True 会把每一步完整打印出来，便于观察闭环
    return AgentExecutor(
        agent=react_agent,
        tools=TOOLS,
        verbose=True,
        memory=memory,               # 多轮记忆（None 则每轮独立）
        handle_parsing_errors=True,  # 模型偶尔格式不规范时，把解析错误回灌让它自我修正
        max_iterations=6,            # 多城市对比可能连续调用多次工具，给足轮次
    )


# ============================================================
# ⑤ 测试用例：覆盖“实时调用成功 / 多轮调用 / 调用失败 / 不调用工具”等路径
# ============================================================
TEST_CASES = [
    ("北京今天天气怎么样？", "调用 get_current_weather，返回高德实时天气"),
    ("上海未来3天的天气预报给我看看", "调用 get_weather_forecast(days=3)，返回逐日预报"),
    ("广州和深圳今天哪里更热？", "连续两次调用 get_current_weather，对比实时气温"),
    ("浙江义乌今天天气怎么样？", "县级市动态解析，验证全国城市实时覆盖"),
    ("你都支持查询哪些城市的天气？", "调用 list_supported_cities，说明全国范围"),
    ("你好，你是谁？", "不调用工具，Agent 直接自我介绍"),
    ("1+1等于多少？", "不调用工具，Agent 直接答 2"),
    ("火星上的天气怎么样？", "地理编码查无此地，Agent 解释只支持中国城市"),
    ("帮我预测一下北京30天后的天气", "days=30 超出 1~7，Agent 解释最多支持未来 7 天"),
]


def run_test_cases(executor: AgentExecutor) -> None:
    """跑固定测试用例（覆盖 实时调用成功 / 多轮调用 / 调用失败 / 不调用工具 路径），汇报截图用。

    每条用例使用独立的执行器（独立记忆），保证用例之间上下文互不污染。
    """
    for idx, (question, expect) in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 70)
        print(f"测试用例 {idx}/{len(TEST_CASES)}：{question}")
        print(f"预期路径：{expect}")
        print("-" * 70)
        try:
            # 每条用例独立一次完整的 ReAct 循环；verbose 日志中可见
            # Thought(思考) -> Action(行动) -> Observation(观察) -> Final Answer(回答)
            case_executor = build_agent_executor(enable_memory=False)
            output = case_executor.invoke({"input": question})
            print(f"\n>>> 最终回答：{output['output']}")
        except Exception as exc:  # 单条用例失败不影响后续用例演示
            print(f"\n>>> 该用例运行出错：{exc}")


def run_chat(executor: AgentExecutor, max_turns: int = 50) -> None:
    """交互模式：用户主动输入问题，Agent 实时回答；输入 q / quit / exit / 退出 结束。"""
    print("\n" + "=" * 70)
    print("我是天气实时查询助手（高德数据），支持全国省/市/区县：")
    print("  - 实时天气 + 未来 7 天预报（1~4 天高德，5~7 天 Open-Meteo）")
    print("  - 中文或拼音输入均可（如 茂名 / maoming）")
    print("  - 本会话支持多轮追问（如“那明天呢？”）")
    print("示例：北京今天天气怎么样？ / 上海未来5天预报 / 广州和深圳哪里更热？")
    print("输入 q / quit / exit / 退出 结束对话")
    print("=" * 70)
    for _turn in range(max_turns):
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):   # Ctrl+C / Ctrl+D 也能安全退出
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"q", "quit", "exit", "退出"}:
            print("再见！")
            break
        try:
            # 每次提问都是一次完整的 ReAct 循环：思考 -> 行动 -> 观察 -> 回答
            output = executor.invoke({"input": question})
            print(f"Agent > {output['output']}")
        except Exception as exc:  # 单轮出错不中断对话，可继续提问
            print(f">>> 该轮运行出错：{exc}")


if __name__ == "__main__":
    # 修复1：Windows GBK 控制台下 LLM 回答可能含 emoji，统一 stdout/stderr 为 UTF-8，
    # 避免 print 抛 UnicodeEncodeError（errors=replace 保证任何字符都不会中断输出）
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    # 运行前检查两个必需的 Key（模型/工具都在导入时不请求，保证本文件也可被 import 做离线单测）
    missing = []
    if not LLM_API_KEY:
        missing.append('  LLM_API_KEY  ：LLM 的 key，如 $env:LLM_API_KEY="sk-你的key"')
    if not AMAP_API_KEY:
        missing.append('  AMAP_API_KEY ：高德「Web服务」key，console.amap.com 免费申请后 '
                       '$env:AMAP_API_KEY="你的key"')
    if missing:
        print("未检测到必需的 Key！请先设置以下环境变量：")
        for line in missing:
            print(line)
        sys.exit(1)

    # 用法：
    #   python main.py          -> 交互模式（默认）：自己输入问题，Agent 实时回答
    #   python main.py --test   -> 只跑固定测试用例（汇报截图用）
    parser = argparse.ArgumentParser(description="天气实时查询助手 Agent（高德数据 + ReAct 工具调用 Demo）")
    parser.add_argument("--test", action="store_true",
                        help="运行固定测试用例；不带本参数则进入交互模式")
    args = parser.parse_args()

    executor = build_agent_executor()

    if args.test:
        run_test_cases(executor)
    else:
        run_chat(executor)
