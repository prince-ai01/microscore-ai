# MicroScore AI

An explainable credit-risk underwriting prototype tailored for informal micro-merchants and street vendors. 

Traditional credit scoring often excludes informal business owners due to a lack of formal credit histories, tax filings, or audited financial statements. **MicroScore AI** bridges this financial inclusion gap by evaluating short-term cash flows, daily transactions, and digital payment metrics to assess repayment viability—complete with transparent, interpretable explanations for every decision.

---

## Key Features

- **Explainable Underwriting:** Utilizes **SHAP (SHapley Additive exPlanations)** values to show exactly how each input feature contributed to an approval or rejection.
- **Risk Assessment & Pricing:** Outputs binary loan decisions alongside calculated approval/default probabilities.
- **Dynamic Credit Limits:** Recommends safe, proportional credit limits calibrated to estimated vendor cash margins.
- **Interactive Web App:** A clean, intuitive [Streamlit](https://streamlit.io/) interface for credit officers or underwriters to test applications.
- **Historical Data Preview:** Inspect prior micro-merchant applications and repayment outcomes directly within the dashboard.

---

## Data Inputs & Features

The model scores applicants using alternative, high-velocity operational signals:

| Feature | Description |
| :--- | :--- |
| **Daily Income** | Average gross revenue generated per working day |
| **Daily Operating Expenses** | Inventory, transit, stall fees, and daily operating costs |
| **Monthly Savings** | Net capital set aside monthly |
| **Daily UPI Transaction Count** | Digital payment frequency (surrogate for customer volume and cash flow visibility) |

---

## Tech Stack

- **Language:** Python
- **Interface:** Streamlit
- **Machine Learning & Explainability:** Scikit-learn, SHAP
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib 

---

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/prince-ai01/microscore-ai.git](https://github.com/prince-ai01/microscore-ai.git)
   cd microscore-ai
