# panda_data — 本 Skill 使用的三个接口

引自 `../../panda_data_api_doc.md`。字段以本文件为准；如与 panda_data 实际返回不一致，
以实际返回为准并回填本文件。

---

## 1. `get_index_weights` — 获取指数权重信息数据

### 1.1 方法名
`get_index_weights`

### 1.2 入参

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| index_symbol | Optional[Union[string, List[string]]] | 指数代码 | 非必填 |
| stock_symbol | Optional[Union[string, List[string]]] | 成分股代码 | 非必填 |
| start_date | string | 开始日期,eg:"20250702" | 必填 |
| end_date | string | 结束日期,eg:"20250702" | 必填 |
| fields | Optional[Union[string, List[string]]] | 返回字段列表 | 非必填 |

### 1.3 响应参数

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| index_symbol | string | 指数代码 |
| date | string | 日期 |
| stock_symbol | string | 股票代码 |
| weight | float | 权重 |

### 1.4 使用示例

```python
import panda_data
result = panda_data.get_index_weights(
    index_symbol="000300.SH",
    stock_symbol="",
    start_date="20260721",
    end_date="20260721",
    fields=None,  # 请求全部列，尤其是 weight
)
print(result)
```

**⚠️ 已知不一致：** 原始文档的示例调用中 `fields=["index_symbol", "stock_symbol", "date"]`
省略了 `weight` 列；响应参数表里 `weight` 是明确的 float 字段。本 Skill 一律 `fields=None`
拉全列，并在 `data.py` 里校验 `weight` 是否真的存在；若缺失，回退到"等权"，日志打 WARN。

---

## 2. `get_stock_daily` — 获取 A 股日线数据

### 2.1 方法名
`get_stock_daily`

### 2.2 入参

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| start_date | string | 开始日期,eg:"20250702"，与结束日期间不超过5年 | 必填 |
| end_date | string | 结束日期,eg:"20250702"，与开始日期间不超过5年 | 必填 |
| symbol | Optional[Union[string, List[string]]] | 股票代码 | 非必填 |
| fields | Optional[Union[string, List[string]]] | 返回字段 | 非必填 |
| indicator | Optional[string] | 股票池（默认为空表示查询所有） | 非必填 |
| st | Optional[bool] | 是否包含 ST 股，默认 True 表示包含 | 非必填 |

### 2.3 响应参数

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 日期 |
| symbol | string | 股票代码 |
| name | string | 股票名称 |
| open | float | 当日开盘价 |
| close | float | 当日收盘价 |
| high | float | 当日最高价 |
| low | float | 当日最低价 |
| volume | float | 当日成交量 |
| amount | float | 当日成交额 |
| pre_close | float | 昨收价 |
| limit_up | float | 当日涨停价 |
| limit_down | float | 当日跌停价 |
| trade_status | integer | 当日是否停牌（0 表示不停牌） |

### 2.4 使用示例

```python
import panda_data
result = panda_data.get_stock_daily(
    symbol=["600519.SH", "000858.SZ"],
    start_date="20260401",
    end_date="20260721",
    fields=["symbol", "date", "close"],
)
print(result)
```

**本 Skill 用途：** 拉取 1 年（约 252 交易日）的 `close`，用于计算：
1. 协方差 Σ 的输入（日收益率）
2. 动量因子分（20 日累计收益）
3. 反转因子分（5 日累计收益）

---

## 3. `get_factor` — 获取回测因子

### 3.1 方法名
`get_factor`

### 3.2 入参

| 字段 | 类型 | 描述 | 是否必填 |
|:---|:---|:---|:---|
| start_date | string | 开始日期,eg:"20250702" | 必填 |
| end_date | string | 结束日期,eg:"20250702" | 必填 |
| symbol | Optional[Union[string, List[string]]] | 股票代码 | 非必填 |
| factors | Union[string, List[string]] | 因子列表 | 必填 |
| type | Optional[string] | 产品类型，支持"stock","future"，默认"stock" | 非必填 |
| index_component | Optional[string] | 股票池（默认为空） | 非必填 |

### 3.3 响应参数（基础因子）

| 字段 | 类型 | 描述 |
|:---|:---|:---|
| date | string | 日期 |
| symbol | string | 标的代码 |
| name | string | 股票名称（仅股票） |
| open | float | 开盘价 |
| close | float | 收盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| volume | float | 成交量 |
| amount | float | 成交额 |
| market_cap | float | 市值（仅股票） |
| turnover | float | 换手率（仅股票） |

> 文档提示：更多因子仅股票类型可用，详见 panda_data 附带的下载文件。本 Skill 只用 `turnover`
> 一个字段，规避未知因子名。

### 3.4 使用示例

```python
import panda_data
result = panda_data.get_factor(
    symbol=["600519.SH", "000858.SZ"],
    start_date="20260401",
    end_date="20260721",
    factors=["turnover"],
    type="stock",
)
print(result)
```

**本 Skill 用途：** 拉取 20 日换手率，取均值作为"换手率因子分"，构造第 3 条 view（换手率高 = 看空）。
