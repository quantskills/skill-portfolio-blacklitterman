# skill-portfolio-blacklitterman

Claude Code skill for a daily Black-Litterman portfolio on 沪深300, driven by three factor views (Momentum / Reversal / Turnover). See `SKILL.md` for the full contract. Design lives in `docs/superpowers/specs/2026-07-29-portfolio-blacklitterman-design.md`; implementation plan in `docs/superpowers/plans/2026-07-29-portfolio-blacklitterman.md`.

## Quick start

```bash
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
pip install -r requirements.txt
pytest tests/                                       # unit tests
python -m scripts.data --self-check --date 20260721 # field self-check
python scripts/portfolio.py --date 20260721         # single-day BL run
```

Outputs land in `output/portfolio_YYYYMMDD.csv` + `.md`.
