# DTU Project 1 — Kidney Stone Urine Analysis

### Data-driven classification of urinary attributes to predict kidney stone-related conditions.

This project implements an end-to-end pipeline—data loading, exploratory analysis, model training, and result visualizations.

---

## Dataset

-   **File:** `kidney_stone_urine_analysis.csv`
-   **Contains:** Urinary attribute measurements (e.g., pH, specific gravity, etc.).
-   **Objective:** Classify samples based on outcomes related to kidney stone diagnostics.

---

## Structure

```
.
├── data/ (optional)
│   └── kidney_stone_urine_analysis.csv
├── notebooks/
│   ├── opgaver.ipynb       # Initial exploration
│   └── plots.ipynb         # Visual analysis
├── src/
│   ├── dataloader.py       # Data loading and preprocessing
│   ├── Boxplot.py          # Distribution plotting
│   ├── attributes_plot.py  # Feature trend visualization
│   └── attributes_plot_correlation.py # Correlation matrix
├── plots/
│   ├── Standardized Boxplot.png
│   └── coefficient_norms.png
├── 02450_project_1.pdf     # Final report
└── README.md
```
---

## Key Steps & Insights

1.  **Exploratory Data Analysis**
    -   Visualized distributions using boxplots to detect outliers and skewed features.
    -   Created correlation matrices to uncover relationships between variables.

2.  **Modeling & Evaluation**
    -   Implemented classification approaches (e.g., decision thresholds informed by data trends).
    -   Interpreted coefficient norms to evaluate feature importance.

3.  **Outputs**
    -   Generated visuals: standardized boxplots and feature coefficient plots.
    -   Compiled a formal report with context, methods, visual results, and interpretations.

---

## How to Run

1.  Ensure Python 3.x is installed.

2.  (Optional) Create a virtual environment:
    ```bash
    python3 -m venv .env
    source .env/bin/activate # or .env\Scripts\activate on Windows
    pip install pandas matplotlib
    ```

3.  Start digging using the notebooks:
    ```bash
    jupyter notebook
    ```

4.  Explore visual scripts:
    ```bash
    python src/Boxplot.py
    python src/attributes_plot.py
    python src/attributes_plot_correlation.py
    ```

---

## Impact & Takeaway

-   Demonstrates ability to handle tabular datasets: clean loading, visual analysis, and feature interpretation.
-   Showcases skills in EDA and visualization—a critical foundation in consulting and data roles.
