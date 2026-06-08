# European Equity Momentum — Asset Pricing with Prof. Gao Can 


## Whats in here

```
quant_research/
├── data/           # price loading + synthetic data generator for testing
├── signals/        # momentum signals (main focus), mean-reversion for comparison
├── backtest/       # simple vectorised backtest, transaction cost aware
├── utils/          # sharpe, drawdown, tearsheet plots etc.
└── example_run.py  # runs everything end to end on synthetic data
```

## Research question

Does the 12-1 month momentum effect (Jegadeesh & Titman 1993) hold in
European equities post-2010? And does vol-scaling (Moskowitz et al. 2012)
improve risk-adjusted performance net of transaction costs?

Results: yes it holds, but weaker after 2018 and TC eats a lot of it
at daily rebalancing. Weekly rebalancing looks better net.
## Quick start

```bash
pip install -r requirements.txt
python example_run.py
```

## Signals implemented

| Signal | Reference | Notes |
|--------|-----------|-------|
| `MomentumSignal` | Jegadeesh & Titman (1993) | 12-1 month, vol-scaled |
| `CrossSectionalMomentum` | Fama & French (1996) | rank-based, top/bottom quintile |
| `MeanReversionSignal` | De Bondt & Thaler (1985) | z-score, added for comparison |
| `RSIMeanReversion` | — | personal experiment, not in the paper |

## Data note

The actual analysis used Refinitiv Eikon data (STOXX Europe 600 constituents).
Loader works with CSV export format. Use `generate_sample_prices()` for synthetic panel.

