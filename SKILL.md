---
name: skill-portfolio-blacklitterman
description: Black-Litterman 组合优化。以沪深300 指数权重为先验，用动量/反转/换手率三条因子视图更新，输出长权重 CSV + Markdown 报告。
tags: [quant, portfolio, black-litterman, optimization]
---

# Black-Litterman 组合优化

## 适用场景
- 想在给定日期基于沪深300 生成一份"相对指数带主观视图"的长权重组合
- 想跟踪三条常见因子视图（动量/反转/换手率）当前把权重推向哪些股票、推走哪些股票
- 想产出可插入下游回测/交易的 CSV 权重表

## 数据接口（panda_data）

| 接口 | 用途 | 关键字段 |
|---|---|---|
| `get_index_weights` | 先验权重 w_prior | `index_symbol, stock_symbol, date, weight` |
| `get_stock_daily` | 1 年日线 → 协方差 Σ + 动量/反转视图 | `symbol, date, close` |
| `get_factor` | 20 日换手率 → 换手率视图 | `symbol, date, turnover` |

字段详见 `references/need_used_api.md`。

## 术语约定

- 动量（20 日累计收益）**高 = 看多** → P 行正号
- 反转（5 日累计收益）**高 = 看空** → 建 P 时翻符号
- 换手率（20 日均）**高 = 看空** → 建 P 时翻符号

首次实测须校准；如反向，用 `--flip_mom` / `--flip_rev` / `--flip_turnover` 逐条翻转，无须改代码。

## 数学（4 步）

```
π      = δ · Σ · w_prior                            (implied returns)
Ω      = diag(τ · P · Σ · Pᵀ)                       (He-Litterman)
μ_bl   = M · [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q],  M = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹
w_raw  = (δΣ)⁻¹ μ_bl,  w_bl = max(w_raw, 0) / sum(...)
```

默认：δ=2.5, τ=0.05, view_return q̂=0.05。所有窗口以 T-1 结尾。

## 视图构造

对每个因子分：
1. 计算截面得分（动量/反转 = 累计收益；换手率 = 均值）。
2. 取 top-decile 与 bottom-decile；构造单位空头-多头行 `P_k`，每行和为 0。
3. Q_k = `view_return`（默认 5%），对三条视图统一。

## 协方差

Ledoit-Wolf shrinkage on 252 交易日日收益，年化 ×252。任意含 NaN 的股票被剔除。

## 优化

闭式解 `w = (δΣ)⁻¹ μ_bl`，负权重截断为 0，重新归一化到 sum=1。
若所有 μ_bl ≤ 0 → 回退至 w_prior，日志打 WARN，仍以 exit=0 收尾。

## 使用方式

```bash
# 认证
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...

# 字段自检
python -m scripts.data --self-check --date 20260721

# 默认扫描
python scripts/portfolio.py

# 指定日期
python scripts/portfolio.py --date 20260721

# 全参数
python scripts/portfolio.py --date 20260721 \
    --index_symbol 000300.SH \
    --delta 2.5 --tau 0.05 --view_return 0.05 \
    --cov_lookback 252 --fetch_days 400 \
    --mom_lookback 20 --rev_lookback 5 --turnover_lookback 20 \
    --min_valid_days 200 \
    --flip_mom --flip_rev --flip_turnover \
    --output_dir output/

# 单元测试
pytest tests/ -v
```

## 输出

**`output/portfolio_YYYYMMDD.csv`**（每只股票一行，按 |Δw| 降序）：

| 列 | 说明 |
|---|---|
| `trade_date` | 扫描日 T |
| `symbol` | 股票代码 |
| `w_prior` | 指数权重（对入围股票归一化后） |
| `w_bl` | BL 长权重 |
| `delta_w` | w_bl − w_prior |
| `pi` | 隐含先验预期收益 (δΣw_prior) |
| `mu_bl` | 后验预期收益 |
| `mu_bl_minus_pi` | 视图对预期收益的推动 |
| `in_view_mom` | +1 / 0 / −1（动量视图中的多/中/空侧，含翻转后） |
| `in_view_rev` | 反转视图归属 |
| `in_view_turnover` | 换手率视图归属 |

**`output/portfolio_YYYYMMDD.md`**：Top 10 超配 / 低配 + 三条视图 top-3 空多列表 + 一句解读；退化路径（回退至 prior）会显式提示。

## 退出码

| Code | 含义 |
|---|---|
| 0 | 成功，CSV + MD 已写 |
| 1 | panda_data 接口 / 认证 / 网络异常 |
| 2 | 扫描日无 get_index_weights 数据 |
| 3 | 过滤后 universe 为空 |
| 4 | 字段自检失败 |
| 5 | LW + jitter 后 Σ 仍非 PSD |

## 验收要求
- **无未来函数**：所有窗口以 T-1 结尾；`test_build_views_no_lookahead_ignores_dates_ge_scan_date` 覆盖
- **单元测试全通过**：`pytest tests/` 无失败
- **He-Litterman 8 国算例**：`test_bl.py` 中的 π 复现误差 <0.6% 绝对值
- **端到端跑通**：mock panda_data 情况下 `test_portfolio_main_end_to_end` 通过；`w_bl.sum()` 在 1 ± 1e-6 内
- **字段自检**：`python -m scripts.data --self-check --date <近期>` 返回 0

## 已知局限（v0.1.0）
- Q̂ 单一全局常数（`--view_return`），未分因子标定；v0.2 计划补上。
- Σ_bl 计算但未用于优化器（详见 spec §6.4）。
- 无交易成本 / 换手率约束。
- Universe 固定 A 股股票指数成分，不接 HK/US/期货。
- 无滚动回测；快照分配器而非研究框架。
- 反转 / 换手率视图的符号约定为通用截面股票配置，可能需 `--flip_*` 校准。
- `get_index_weights` 文档示例调用未取 `weight`；实测若缺失 → 回退等权 + WARN。
