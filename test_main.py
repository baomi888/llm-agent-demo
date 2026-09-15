# -*- coding: utf-8 -*-
"""
main.py 离线单元测试（优化6）：全部 HTTP 请求被 mock，不访问网络、不消耗任何 API 配额。

运行方式：
    python -m pytest test_main.py -v

覆盖范围：
    - 地名解析：别名归一化、解析成功、模糊误匹配拦截（火星）、纯字母拼音分支
    - 工具层：实况格式化、1~4 天高德预报、5~7 天 Open-Meteo 切换、days 越界校验
    - ReAct 解析器：多参数 JSON Action Input、普通字符串输入
    - 提示词：必需占位符齐全（含多轮记忆的 chat_history）
"""

import json

import main
import pytest


# ---------- 测试辅助：伪造高德/Open-Meteo 响应 ----------

def _fake_amap_geocode(address="茂名市", adcode="440900", location="110.925,21.663",
                       province="广东省", city="茂名市", district=""):
    """构造高德地理编码接口的标准返回结构。"""
    return {"status": "1", "geocodes": [{
        "province": province, "city": city, "district": district,
        "adcode": adcode, "location": location, "level": "市",
    }]}


def _fake_amap_live(weather="多云", temperature="30", humidity="69",
                    winddirection="东风", windpower="≤3", reporttime="2026-09-15 12:00:00"):
    """构造高德实况接口的标准返回结构。"""
    return {"status": "1", "lives": [{
        "weather": weather, "temperature": temperature, "humidity": humidity,
        "winddirection": winddirection, "windpower": windpower, "reporttime": reporttime,
    }]}


def _fake_amap_forecast(days=4):
    """构造高德预报接口的标准返回结构（含今天的 days 天）。"""
    casts = [{"date": f"2026-09-{15 + i}", "dayweather": "多云", "nightweather": "阴",
              "daytemp": str(28 + i), "nighttemp": str(20 + i),
              "daywind": "东", "daypower": "1-3"} for i in range(days)]
    return {"status": "1", "forecasts": [{"city": "茂名市", "casts": casts}]}


@pytest.fixture(autouse=True)
def _clear_cache():
    """每条用例前后清空进程内缓存，保证用例互不影响。"""
    main._ADCODE_CACHE.clear()
    yield
    main._ADCODE_CACHE.clear()


# ---------- 地名解析 resolve_city ----------

def test_alias_pinyin_normalized_before_geocode(monkeypatch):
    """拼音别名应先归一化为中文再请求地理编码（maoming -> 茂名）。"""
    seen = {}

    def fake_http(url, params, **kw):
        seen["address"] = params.get("address")
        return _fake_amap_geocode()

    monkeypatch.setattr(main, "_http_get_json", fake_http)
    std, adcode, _loc = main.resolve_city("maoming")
    assert seen["address"] == "茂名"
    assert std == "茂名" and adcode == "440900"


def test_resolve_city_returns_location(monkeypatch):
    """解析成功时应同时返回经纬度（供 Open-Meteo 使用）。"""
    monkeypatch.setattr(main, "_http_get_json",
                         lambda url, params, **kw: _fake_amap_geocode())
    result = main.resolve_city("茂名")
    assert result == ("茂名", "440900", "110.925,21.663")


def test_resolve_city_mismatch_returns_none(monkeypatch):
    """模糊误匹配应被拦截：查询“火星”但高德返回“黔东南…”不含“火星” -> None。"""
    monkeypatch.setattr(main, "_http_get_json", lambda url, params, **kw:
                        _fake_amap_geocode(address="黔东南苗族侗族自治州",
                                           province="贵州省", city="黔东南苗族侗族自治州"))
    assert main.resolve_city("火星") is None


def test_resolve_city_pinyin_miss_returns_none(monkeypatch):
    """纯字母输入且拼音索引查不到（如 huoxing）-> None，且不请求地理编码。"""

    def fail_http(url, params, **kw):  # 地理编码不应被调用
        raise AssertionError("纯拼音未命中时不应请求地理编码接口")

    monkeypatch.setattr(main, "_http_get_json", fail_http)
    monkeypatch.setattr(main, "_load_pinyin_index", lambda: {})
    assert main.resolve_city("huoxing") is None


# ---------- 工具层 get_current_weather ----------

def test_current_weather_formatting(monkeypatch):
    """实况工具应输出天气/气温/湿度/风力及数据发布时间。"""
    monkeypatch.setattr(main, "resolve_city",
                         lambda name: ("茂名", "440900", "110.9,21.6"))
    monkeypatch.setattr(main, "_fetch_live_weather", lambda adcode: {
        "weather": "多云", "temperature": "30", "humidity": "69",
        "winddirection": "东风", "windpower": "≤3", "reporttime": "2026-09-15 12:00:00"})
    out = main.get_current_weather.invoke({"city": "茂名"})
    assert "茂名当前多云" in out and "30℃" in out and "2026-09-15 12:00:00" in out


def test_current_weather_city_not_found(monkeypatch):
    """地名查不到时应返回明确错误信息而不是抛异常（错误信息作为 Observation）。"""
    monkeypatch.setattr(main, "resolve_city", lambda name: None)
    out = main.get_current_weather.invoke({"city": "火星"})
    assert out.startswith("错误：") and "未找到" in out


# ---------- 工具层 get_weather_forecast ----------

def test_forecast_days_validation_without_network():
    """days 越界应在触网前被拦截（30 超出 1~7）。"""
    out = main.get_weather_forecast.invoke({"city": "北京", "days": 30})
    assert out.startswith("错误：") and "1~7" in out


def test_forecast_days_bool_normalized():
    """days=True（bool 是 int 子类）会在 LangChain 的 pydantic 参数校验层被归一化为 1，
    不会越界也不会崩溃——验证其行为等价于合法的 days=1（类型边界在 schema 层收口）。"""
    out = main.get_weather_forecast.invoke({"city": "北京", "days": True})
    assert "未来 1 天" in out


def test_forecast_amap_within_4_days(monkeypatch):
    """1~4 天走高德，逐日输出并标注高德来源。"""
    monkeypatch.setattr(main, "resolve_city", lambda name: ("茂名", "440900", "110.9,21.6"))
    monkeypatch.setattr(main, "_fetch_forecast", lambda adcode: _fake_amap_forecast()["forecasts"][0]["casts"])
    out = main.get_weather_forecast.invoke({"city": "茂名", "days": 3})
    assert "未来 3 天" in out and "高德天气预报数据" in out and "9月17日" in out


def test_forecast_open_meteo_beyond_4_days(monkeypatch):
    """5~7 天应切换 Open-Meteo，且不再调用高德预报接口。"""
    monkeypatch.setattr(main, "resolve_city", lambda name: ("茂名", "440900", "110.925,21.663"))
    called = {}

    def fake_open_meteo(location, days):
        called["location"], called["days"] = location, days
        return [{"date": "2026-09-20", "day": "小雨", "max": 28, "min": 21},
                {"date": "2026-09-21", "day": "阴", "max": 27, "min": 20}]

    def fail_amap_forecast(adcode):  # 5~7 天不应请求高德预报
        raise AssertionError("days>4 时不应请求高德预报接口")

    monkeypatch.setattr(main, "_fetch_open_meteo_daily", fake_open_meteo)
    monkeypatch.setattr(main, "_fetch_forecast", fail_amap_forecast)
    out = main.get_weather_forecast.invoke({"city": "茂名", "days": 7})
    assert called == {"location": "110.925,21.663", "days": 7}
    assert "Open-Meteo" in out and "小雨" in out


def test_forecast_missing_location_rejected(monkeypatch):
    """地理编码未返回经纬度时，5~7 天预报应如实报错。"""
    monkeypatch.setattr(main, "resolve_city", lambda name: ("某城", "440900", ""))
    out = main.get_weather_forecast.invoke({"city": "某城", "days": 6})
    assert out.startswith("错误：") and "经纬度" in out


# ---------- ReAct 输出解析器 ----------

def test_parser_json_action_input():
    """多参数 JSON 的 Action Input 应被解析成 dict，供工具按具名参数绑定。"""
    text = ('Thought: 需要预报\nAction: get_weather_forecast\n'
            'Action Input: {"city": "北京", "days": 3}')
    action = main.ReActJsonArgsOutputParser().parse(text)
    assert action.tool == "get_weather_forecast"
    assert action.tool_input == {"city": "北京", "days": 3}


def test_parser_json_fence_tolerated():
    """模型偶尔用 ```json 包裹 Action Input 时应容错解析。"""
    text = ('Action: get_current_weather\n'
            'Action Input: ```json\n{"city": "上海"}\n```')
    action = main.ReActJsonArgsOutputParser().parse(text)
    assert action.tool_input == {"city": "上海"}


def test_parser_plain_string_kept():
    """非 JSON 的单字符串输入应原样保留（向后兼容单参数工具）。"""
    text = "Action: get_current_weather\nAction Input: 北京"
    action = main.ReActJsonArgsOutputParser().parse(text)
    assert action.tool_input == "北京"


def test_parser_final_answer_passthrough():
    """Final Answer 应解析为 AgentFinish 且不做 JSON 处理。"""
    from langchain_core.agents import AgentFinish
    text = "Thought: 知道了\nFinal Answer: 今天晴天"
    result = main.ReActJsonArgsOutputParser().parse(text)
    assert isinstance(result, AgentFinish)


# ---------- 提示词与静态配置 ----------

def test_prompt_has_required_placeholders():
    """ReAct 提示词必须包含全部必需占位符（含多轮记忆的 chat_history）。"""
    need = {"tools", "tool_names", "input", "agent_scratchpad", "chat_history"}
    assert need.issubset(set(main.REACT_PROMPT.input_variables))


def test_wmo_code_map_covers_common():
    """WMO 天气码表应覆盖常见码值（Open-Meteo 5~7 天预报依赖它转中文）。"""
    for code in (0, 1, 2, 3, 45, 51, 61, 63, 65, 71, 80, 95, 99):
        assert main._WMO_CN.get(code)


def test_http_get_json_open_meteo_no_key_needed(monkeypatch, tmp_path):
    """Open-Meteo（amap_status=False）不应附带高德 key，也不校验 status 字段。"""
    captured = {}

    class FakeResp:  # 最小化上下文管理器，模拟 urlopen 返回
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"daily": {"time": ["2026-09-16"]}}).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return FakeResp()

    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)
    data = main._http_get_json(main._OPEN_METEO_URL, {"latitude": "21.6"}, amap_status=False)
    assert "key=" not in captured["url"]          # 不带高德 key
    assert data["daily"]["time"] == ["2026-09-16"]  # 不校验 status 也返回
