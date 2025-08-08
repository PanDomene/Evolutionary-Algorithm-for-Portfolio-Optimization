# 🧬 Evolutionary Portfolio Optimization with Risk Constraints

This project implements an **evolutionary algorithm (EA)** for **portfolio optimization**, designed to find asset allocations that maximize return under **realistic constraints** like maximum volatility and weight per asset. It includes full benchmarking against traditional portfolio strategies.

---

## 📁 Project Structure

```
.
├── data/                      # Input datasets (adjusted close prices)
├── EA_funcs/                 # Modular EA package
│   ├── __init__.py
│   ├── algorithms.py         # EA class and benchmarks
│   ├── data.py               # Data loading & preparation
│   ├── metrics.py            # Performance evaluation
│   ├── testing.py            # Robustness checks
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── benchmarks.ipynb      # Strategy comparison & performance
├── results/
│   ├── results.ipynb         # Final visualizations and discussion
├── requirements.txt
├── setup.py                  # To install EA_funcs as a package
└── README.md
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the `EA_funcs` package

```bash
pip install -e .
```

### 4. Download the data

The project automatically fetches historical daily closing prices for a predefined universe (e.g., S&P 100):

```bash
python -m EA_funcs.data
```

This will save the dataset to `data/closing_prices.csv`.

---

## 🚀 Quick Start (Python)

```python
from EA_funcs.data import load_data
from EA_funcs.algorithms import EA

# Load the dataset
data = load_data("data/closing_prices.csv")

# Initialize and run the EA optimizer
ea = EA(data, max_w=0.1, max_risk=0.2)
ea.run(iters=100)

# Display best portfolio
print(ea.portfolio())
```

---

## 📊 Benchmarking

Run `notebooks/benchmarks.ipynb` to compare the EA optimizer with:

- 📏 Equal Weight (EW)
- 🔁 Inverse Volatility
- 🧮 Minimum Variance (long-only)
- 🌲 Hierarchical Risk Parity
- 🎯 Random constrained portfolios

Each strategy is implemented in `EA_funcs.algorithms`.

---

## 🛠 EA Parameters

| Argument         | Description                          | Default |
|------------------|--------------------------------------|---------|
| `pop_size`       | Population size                      | 100     |
| `lambda_`        | Offspring per generation             | 600     |
| `p_m`            | Mutation rate per gene               | 0.01    |
| `sigma`          | Mutation step size                   | 0.1     |
| `delta`          | Return-risk tradeoff weight          | 0.5     |
| `alpha`          | BLX-α crossover parameter            | 0.5     |
| `max_w`          | Max weight per asset                 | 0.1     |
| `max_risk`       | Max allowed portfolio volatility     | None    |
| `validation_split` | Data split for validation          | 0.3     |

---

## 🧾 Key Results

The evolutionary algorithm consistently outperformed benchmark strategies in cumulative returns while satisfying strict constraints on volatility and maximum weight per asset. Specifically:

- ⚖️ Achieved **higher out-of-sample cumulative returns** than traditional strategies like Minimum Variance and Equal Weight, beating the benchmark on 91% of runs.
- 📉 Maintained **risk below the specified `max_risk` threshold**, validating the effectiveness of the volatility constraint.
- 🧬 Showed **stable convergence patterns** across runs, with meaningful population diversity and consistent fitness improvements.

See `results/results.ipynb` for detailed performance metrics, plots, and allocation visualizations.

---

## 🧪 Robustness & Testing Highlights

Several experiments were conducted to test the robustness of the optimizer:

- ✅ **Repeatability across random seeds:** Consistent final performance with low variance in outcomes.
- 📉 **Constraint adherence:** All portfolios respected the `max_w` (max asset weight) and `max_risk` (max volatility) limits.
- 🔍 **Ablation testing:** EA performance degraded gracefully when components like mutation or diversity were restricted or amplified, confirming diversification is the better strategy, if done inmoderation.
- 🌐 **Robustness to Seach Space:** Outperformed the EW portfolio for different asset collections in 92 out of 100 runs, achieving an average relative increase of 18.7% in returns and 3.4% in risk compared to the EW portfolio. 

All robustness tools and tests are implemented in `EA_funcs.testing` and documented in the `results/results.ipynb` notebooks.


## 🔮 Future Work

- Support multi-objective optimization (e.g., Pareto front)
- Incorporate transaction cost modeling
- Add dynamic rebalancing and walk-forward validation
- Streamlit dashboard for interactive use

---

## 📜 License

Feel free to use, fork, and contribute!

---

## 👤 Author

**Juan Domene Ashida**  

📍 Guadalajara, MX
🧠 Physicist & MDS Candidate
🍞 Co-founder at a local bakery business
💼 Aspiring Data Scientist

---
