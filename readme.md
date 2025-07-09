# 🧬 Evolutionary Algorithm for Portfolio Optimization

This project implements an **evolutionary algorithm** to optimize a financial investment portfolio, aiming to **maximize expected annual returns** while keeping risk under control. The model balances **risk and return** through a flexible and customizable framework inspired by Markowitz's mean-variance model, enhanced with evolutionary computation strategies.

## 📈 Objective

- **Maximize portfolio returns** using a real-coded evolutionary algorithm.
- **Limit risk** based on historical volatility.
- **Ensure diversification** by setting constraints on the proportion invested in each asset.
- **Compare portfolio performance** against real-world financial benchmarks (e.g., Vanguard ETFs).

## 🧠 Methodology

### 1. 📈 Data Collection

- Historical closing prices from **Yahoo Finance** API.
- Assets include:
  - Mexican and international stocks traded on the BMV and SIC.
  - Government instruments: **CETES**, **BONDDIA**, and **ENERFIN**.
- Timeframe: **January 2021 – April 2024**.

### 2. 🧮 Fitness Function

```
f(w) = (1 - δ) * Return(w) - δ * Risk(w)
```

- `δ` is a risk aversion parameter (0 ≤ δ ≤ 1).
- `Return(w)` is the expected annual return.
- `Risk(w)` is the annualized portfolio volatility (from the covariance matrix).
- **Penalties** apply when allocation limits are violated.

### 3. 🧬 Evolutionary Algorithm

- **Chromosome**: Real-valued vector of asset weights.
- **Operators**:
  - Selection: Uniform for parents, adaptive tournament for survivors.
  - Recombination: BLX-α crossover.
  - Mutation: Gaussian perturbation.
- **Constraints**:
  - Sum of weights = 1
  - Each weight ≤ `w_max` (default 0.07)

## 🔧 Parameters Used

| Parameter           | Value |
| ------------------- | ----- |
| Population size (μ) | 1000  |
| Offspring (λ)       | 6000  |
| Mutation rate       | 0.05  |
| Mutation std dev    | 0.1   |
| Recombination α     | 0.5   |
| Risk aversion δ     | 0.51  |
| Max asset weight    | 0.07  |

## 📈 Results

- ✅ Outperformed **VMEX ETF** benchmark in **76.7%** of runs.
- 📈 Average return: **\~64.5%**
- ⚠️ Average risk: **\~25.7%**
- The best portfolio: **45.78% return** with **13.5% risk**.

## 📌 Key Insights

- A moderate δ (\~0.5) offers the best trade-off between return and risk.
- Excessive diversification (very low `w_max`) reduces performance.
- The algorithm is robust to different subsets of assets and time periods.

## 🧪 Evaluation Metrics

- **MBF (Mean Best Fitness)**
- **SR (Success Rate)** – % of runs that beat the benchmark.
- **AES (Average Evaluations to Success)**

## 🧰 Tech Stack

- Python
- NumPy, pandas, matplotlib
- `yfinance` for data collection

## 📁 Project Structure

```
portfolio-optimization/
├── data/
├── notebooks/
├── src/
├── results/
├── requirements.txt
├── README.md
└── README_es.md
```

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script or notebook:

```bash
python src/main.py
# or
jupyter notebook notebooks/portfolio_analysis.ipynb
```

## 📌 Future Work

- Hyperparameter tuning with grid/random search.
- Real-time rebalancing strategies.
- Extension to other asset classes and international markets.
- Deploying a dashboard for visualization.

## 📚 References

- Markowitz, H. (1952). Portfolio Selection.
- Eiben & Smith (2015). Introduction to Evolutionary Computing.
- Yahoo Finance API via `yfinance`

