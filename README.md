# Early Detection of Parkinson's Disease Using Voice Features

A statistical-learning-driven study of whether a small set of acoustic voice measurements can reliably flag early signs of Parkinson's disease. The project is deliberately interpretability-first: every preprocessing choice, assumption check, and model variant is justified from the data, not picked for raw accuracy.

## Why This Project

Voice impairment — jitter, shimmer, reduced harmonic-to-noise ratio — is one of the earliest observable symptoms of Parkinson's disease, often preceding motor symptoms by years. A low-cost model that screens voice recordings could support earlier clinical follow-up, which is exactly where outcomes for Parkinson's patients improve the most. The goal here is not to replace diagnosis; it is to understand which voice features carry signal, whether that signal survives rigorous statistical assumptions, and how model choices change the story.

## Approach

The notebook walks through five phases, each grounded in statistical reasoning rather than model-zoo churn:

1. **Exploratory Data Analysis** — variable tagging, distributional shape, skewness/kurtosis, Shapiro–Wilk normality, QQ plots, outliers, missingness.
2. **Feature Relationships & Redundancy** — Pearson/Spearman correlation, VIF for multicollinearity, PCA for dimensionality insight.
3. **Bivariate & Class Comparison** — box plots and ECDFs by Parkinson's status, parametric and non-parametric group tests.
4. **Pre-Model Checks & Assumption Validation** — scaling, log transforms, Box–Tidwell test for linearity of the logit, independence, class balance, SMOTE.
5. **Interpretable Modeling** — logistic regression under five preprocessing configurations, plus a decision tree for transparent rule-based comparison.

Rather than reporting a single "best" model, the project benchmarks **five logistic-regression settings** that each address a specific statistical concern, so the reader can see exactly how preprocessing decisions trade off accuracy, recall, and inferential stability.

## Dataset

The dataset (`data/parkinsons.data`) is the UCI **Parkinsons Data Set**, contributed by Max Little (University of Oxford):

- **195** voice recordings from **31 subjects** (23 with Parkinson's disease)
- **22 acoustic features** per recording: pitch (`MDVP:Fo/Fhi/Flo`), jitter (`MDVP:Jitter`, `Jitter:DDP`, …), shimmer, noise-to-harmonics ratios (`NHR`, `HNR`), nonlinear dynamical complexity (`RPDE`, `D2`), signal fractal scaling (`DFA`), and nonlinear fundamental frequency variation (`spread1`, `spread2`, `PPE`)
- **Target:** `status` — 1 = Parkinson's, 0 = healthy

Source: https://archive.ics.uci.edu/ml/datasets/parkinsons

## Results

Logistic regression was evaluated across five preprocessing settings. All metrics are on the held-out test split.

| Setting                                       | Train Acc. | CV Acc. | Test Acc. | Precision | Recall | F1    | ROC AUC |
| --------------------------------------------- | ---------- | ------- | --------- | --------- | ------ | ----- | ------- |
| Raw Data                                      | 85.90%     | 81.46%  | **92.31%** | 93.33%    | 96.55% | 94.92% | 0.890  |
| Standardized                                  | 86.54%     | 82.12%  | **92.31%** | 93.33%    | 96.55% | 94.92% | 0.779  |
| Log + Standardized                            | 87.18%     | 85.25%  | 89.74%    | 90.32%    | 96.55% | 93.33% | 0.810  |
| SMOTE + Standardized                          | 86.86%     | 84.78%  | 76.92%    | 95.45%    | 72.41% | 82.35% | 0.752  |
| Log + Standardized + SMOTE                    | 88.56%     | 86.05%  | 82.05%    | 92.31%    | 82.76% | 87.27% | 0.903  |
| SMOTE + Redundant Removal + Standardized      | 86.44%     | 84.35%  | 74.36%    | 91.30%    | 72.41% | 80.77% | **0.921** |

**Key takeaways:**

- The raw-data and standardized-only settings produce the highest test accuracy and F1, suggesting the unprocessed data already carries strong linear signal for Parkinson's detection.
- SMOTE improves ROC AUC and coefficient stability but drops test accuracy — a classic bias/variance trade.
- Log transforms help with skewness and assumption validity but do not translate to headline-accuracy gains here.
- The **decision tree** (see notebook, Section 5.2) is included as a transparent, rule-based alternative for clinical interpretability.

The full inferential story — coefficient sign flips, odds ratios, p-values, Wald test, and Box–Tidwell diagnostics — is documented inline in the notebook.

## Repository Structure

```
Parkinsons-Voice-Detection/
├── data/
│   └── parkinsons.data                          # UCI Parkinsons dataset
├── notebooks/
│   └── parkinsons_voice_analysis.ipynb          # Full statistical-learning workflow
├── reports/
│   ├── statistical_learning_report.pdf          # Written report
│   └── statistical_learning_slides.pdf          # Presentation deck
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- `pip` or `conda`

### Setup

```bash
git clone https://github.com/Wobbly1212/Parkinsons-Voice-Detection.git
cd Parkinsons-Voice-Detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter lab notebooks/parkinsons_voice_analysis.ipynb
```

The notebook loads `../data/parkinsons.data`, so launch it from the `notebooks/` directory (Jupyter does this by default).

## Tech Stack

- **Data & numerics:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical testing:** `scipy.stats`, `statsmodels` (Shapiro–Wilk, Box–Tidwell, VIF, Wald test)
- **Modeling:** `scikit-learn` (logistic regression, decision tree, PCA, cross-validation)
- **Class balancing:** `imbalanced-learn` (SMOTE)

## Dataset Citation

Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007). *Suitability of dysphonia measurements for telemonitoring of Parkinson's disease.* BioMedical Engineering OnLine.

## Author

**Diako Darabi** — M.Sc. Data Science
GitHub: [@Wobbly1212](https://github.com/Wobbly1212)

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
