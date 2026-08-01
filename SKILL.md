---
name: skill-portfolio-blacklitterman
description: Black-Litterman 组合优化 —— 用户问「跑一下 BL 组合」「基于视图的组合权重」「相对沪深300 的主动配置」「动量/反转/换手视图对权重的影响」类问题时触发。以沪深300 指数权重为先验，用动量/反转/换手率三条因子视图更新，输出长权重组合，按「样式② 结构化播报」呈现给用户。
tags: [quant, portfolio, black-litterman, optimization]
---

# Black-Litterman 组合优化

## 何时触发本 skill

用户提问命中下列语义时，自动调用：

- 「跑一下 BL / Black-Litterman 组合」「BL 优化」
- 「基于视图的组合权重」「今日/YYYYMMDD 的主动配置」
- 「动量/反转/换手视图现在指向哪些股票」
- 「相对沪深300 应该超配/低配什么」
- 「三条因子视图今天给出什么方向」

**不触发**：单只股票的多空判断、无视图的等权/市值配置、期货/港美股组合（本 skill 仅 A 股沪深300 成分）。

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

## Agent 触发流程（本 skill 的正式用法）

用户提问命中「何时触发」后，按四步执行，**不要跳步、不要问用户参数**：

### Step 1 · 决定扫描日期

- 用户明说了日期 → 换算为 `YYYYMMDD` 用作 `--date`
- 用户没说 → 省略 `--date`，让 portfolio 自动取最近可用交易日
- 用户说"最近"、"这一周" → 仍按单日跑（本 skill 是快照分配器，不做滚动）

### Step 2 · 调用（推荐一行）

```bash
cd /Users/since/Code/quantskills/skill-portfolio-blacklitterman && \
set -a && source ~/.zshrc >/dev/null 2>&1 && set +a && \
/opt/miniconda3/envs/pandaai/bin/python scripts/portfolio.py [--date YYYYMMDD]
```

- 环境是 conda `pandaai`（Python 3.10，`panda_data` 已装）
- 凭证 `PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD` 在 `~/.zshrc`（非交互 shell 须显式 source）
- 默认参数 `delta=2.5, tau=0.05, view_return=0.05`，v0.1.0 首次校准已确认**三视图默认符号方向都对**，无需 `--flip_*`
- exit code：0 OK / 1 panda_data 异常 / 2 该日无指数权重 / 3 池空 / 4 字段自检失败 / 5 Σ 非 PSD

### Step 3 · 读取输出

产物固定在两个位置：

- `output/portfolio_YYYYMMDD.csv` —— 每股一行的全量权重表，字段见「输出」
- `output/portfolio_YYYYMMDD.md` —— Top 10 超配 / 低配 + 三视图 top-3 + 一句解读

**直接读 `.md`** 拿排行，需要权重绝对值、`mu_bl` 或 `in_view_*` 明细再看 `.csv`。

### Step 4 · 用「样式② 结构化播报」呈现

**不要**把 CSV 路径丢给用户，也**不要**贴 markdown 原文。按固定六段呈现：

```
BL 组合优化 · 沪深300 · YYYYMMDD（池 N 只）

▎主线判断：<一句话，见下表>

▎最大超配（关注视图共振的加仓方向）
- <symbol>：Δw +X.XX%（w_prior X.XX% → w_bl X.XX%），active <如 +MOM +TUR>
- <symbol>：Δw +X.XX%，active <...>
- <symbol>：Δw +X.XX%，active <...>

▎最大低配（关注视图共振的减仓方向）
- <symbol>：Δw -X.XX%（w_prior X.XX% → w_bl X.XX%），active <如 -MOM -TUR>
- <symbol>：Δw -X.XX%
- <symbol>：Δw -X.XX%

▎视图诊断（三视图分别把权重推向哪里）
- 动量：多 <top3 symbol> · 空 <top3 symbol>
- 反转：多 <top3> · 空 <top3>
- 换手：多 <top3> · 空 <top3>

▎组合特征
- 相对指数换手 |Δw|.sum/2 ≈ XX.XX%
- long-only 裁剪：M 只被截到 0
- w_bl.sum 校验：1.00000... （若非 1e-6 内须显式提示）
```

**主线判断话术表**：

| 场景 | 话术 |
|---|---|
| 视图强共振（top3 超配/低配都是三视图重叠） | 「三视图共振，明确方向配置：加 <行业主题> / 减 <行业主题>」 |
| 视图分歧（超配和低配的视图标签零散） | 「三视图分歧，主要靠单一视图推动，方向弱信号」 |
| 换手 ≤ 5% | 「视图对权重推动微弱，组合接近指数复制」 |
| 退化路径（回退至 w_prior，output MD 有"degenerate"提示） | 「所有后验预期收益非正，已回退至指数权重（视图与协方差冲突）」 |

**数据侧特殊情况**（Agent 必须显式说明）：

- **`get_index_weights` 未取到 weight → 回退等权**（output MD 会有 WARN） → 呈现时须写："先验权重回退至等权（`get_index_weights` 未返回 weight 字段），组合结果需谨慎"
- **exit 5（Σ 非 PSD）** → "协方差矩阵在 Ledoit-Wolf + jitter 后仍非正定，无法优化，请换日期或调 `--cov_lookback`"
- **exit 3（池空）** → "该日过滤后无满足条件的成分股，可能是非交易日或数据缺失"

**行业归因（可选，需读 CSV 时）**：如果用户追问"为什么加这些",看 `active views` 列里最频繁出现的组合（+MOM +TUR 通常意味着"高动量 + 低换手"，即被市场稳定推动的价值股）。

**收尾一句**（可选）：如果用户看起来还会追问，加"如需看具体股票、翻转视图符号、或换日期，告诉我"。

## 参数调整（用户主动要求时才调）

用户明确说要翻符号、调 `view_return`、换指数等，透传对应 CLI 参数：

```bash
python scripts/portfolio.py --date YYYYMMDD \
    --index_symbol 000300.SH \
    --delta 2.5 --tau 0.05 --view_return 0.05 \
    --cov_lookback 252 --fetch_days 400 \
    --mom_lookback 20 --rev_lookback 5 --turnover_lookback 20 \
    --min_valid_days 200 \
    --flip_mom --flip_rev --flip_turnover \
    --output_dir output/
```

否则一律用默认阈值。

## 开发者入口（不用于 Agent 触发路径）

```bash
# 字段自检（升级 panda_data 后手动跑一次）
python -m scripts.data --self-check --date 20260729

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
