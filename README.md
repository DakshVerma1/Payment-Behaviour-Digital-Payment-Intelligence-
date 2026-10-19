# Payment Behaviour & Digital Payment Intelligence

**Course:** Fundamentals of Machine Learning (MLE2503)  
**Programme:** B.Tech AI & ML, Semester 5  
**Project type:** Unsupervised machine learning / customer segmentation

## Overview

This project explores whether customers can be grouped according to their payment preferences and transaction behaviour. It compares segmentation based on payment-mode ratios alone with segmentation that also uses spending, transaction frequency, cashback/discount use, EMI use and channel behaviour.

The workflow covers data inspection, cleaning, customer-level feature engineering, standardization, K-Means clustering, Agglomerative clustering, internal evaluation metrics, PCA visualization and business recommendations.

## Dataset

The included `synthetic_payment_transactions.csv` is **synthetic data created for demonstration**. The assignment document supplied for this project describes the problem and required tasks but does not include transaction-level data. Results in this repository therefore do not represent real customers, a real payment provider or the Indian payment market.

If an instructor-approved dataset becomes available, update the input data and rerun the analysis before using the results in a report or presentation.

### Main transaction fields

- `transaction_id`: transaction identifier
- `customer_id`: customer identifier
- `transaction_date`: transaction date
- `payment_mode`: cash, card, UPI, wallet or EMI/BNPL
- `transaction_amount`: transaction value
- `cashback_used`: whether cashback was used
- `discount_percent`: discount percentage
- `emi_used`: whether EMI was used
- `channel`: app, website or physical store

## Project structure

```text
Payment_Behaviour_Digital_Payment_Intelligence/
├── payment_behaviour_project.py
├── Payment_Behaviour_Project.ipynb
├── synthetic_payment_transactions.csv
├── customer_level_features.csv
├── customer_segments.csv
├── cluster_profiles.csv
├── model_comparison.csv
├── requirements.txt
└── plots/
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**macOS/Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

Run the Python script:

```bash
python payment_behaviour_project.py
```

The script reads `synthetic_payment_transactions.csv`, performs preprocessing and clustering, and generates/updates the customer-level CSV outputs and PNG plots.

You can also open `Payment_Behaviour_Project.ipynb` in Jupyter or Google Colab and run its cells from top to bottom. Ensure the CSV is in the notebook's working directory.

## Methods used

- **Pandas:** data loading, inspection, cleaning and aggregation
- **StandardScaler:** feature scaling before distance-based clustering
- **K-Means:** primary clustering method
- **Agglomerative clustering:** hierarchical comparison
- **Elbow Method:** inspect inertia across candidate cluster counts
- **Silhouette Score:** assess cluster separation (higher is generally better)
- **Davies–Bouldin Index:** assess cluster compactness/separation (lower is generally better)
- **PCA:** two-dimensional visualization of cluster patterns

## Important interpretation note

Cluster numbers are arbitrary. Interpret each cluster using the profile table rather than treating a cluster ID as a customer type by itself. Internal metrics are not enough to prove business usefulness; interpretability, stability and practical relevance also matter.

## Limitations

- The dataset is synthetic.
- Clustering is exploratory and does not establish causation.
- This project does not predict fraud or creditworthiness.
- Real-world use would require approved representative data, privacy safeguards, stability checks and business validation.

## References

1. Amity School of Engineering and Technology, *Assignment, July–Dec 2026: Fundamentals of Machine Learning (MLE2503), Problem Statement 7: Payment Behaviour & Digital Payment Intelligence*.
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. pandas documentation: https://pandas.pydata.org/docs/
