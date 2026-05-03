# ML Comparative Analysis — Heart Disease & Student Performance

A Streamlit web application that performs a comparative analysis of Machine Learning models for two supervised learning tasks: classification (heart disease prediction) and regression (student math score prediction).

---

## Project Structure

```
├── streamlit_app.py                  # Main Streamlit application
├── heart.csv                         # Heart disease dataset (UCI)
├── Original_data_with_more_rows.csv  # Student exam scores dataset (Kaggle)
├── notebook_clasificare.ipynb        # Classification analysis notebook
└── notebook_regresie.ipynb           # Regression analysis notebook
```

---

## Tasks

### Classification — Heart Disease Prediction
- **Dataset:** UCI Heart Disease (1,102 patients)
- **Target:** Presence of heart disease (binary)
- **Features:** Age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and more
- **Models:** Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, CatBoost, EBM

### Regression — Student Math Score Prediction
- **Dataset:** Students Exam Scores — Kaggle (30,641 students)
- **Target:** Math score (continuous, 0–100)
- **Features:** Gender, ethnic group, parent education, lunch type, test preparation, reading & writing scores
- **Models:** Linear Regression, Decision Tree, Random Forest, SVR, KNN, XGBoost, CatBoost, EBM

---

## Pipeline

For both tasks the app follows the same pipeline:

1. **EDA** — distributions, correlations, boxplots
2. **Preprocessing** — one-hot encoding, train/test split (75/25), StandardScaler
3. **Base model training** — all models with default hyperparameters
4. **Hyperparameter tuning** — GridSearchCV with 5-fold CV on top 5 models
5. **Evaluation** — metrics, learning curves, confusion matrix / predictions vs actual
6. **Prediction** — interactive input form with SHAP explanation (waterfall plot)

---

## Installation

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost catboost interpret shap
```

---

## Usage

Place `heart.csv` and `Original_data_with_more_rows.csv` in the same directory as `streamlit_app.py`, then run:

```bash
streamlit run streamlit_app.py
```

---

## Notes

- The heart disease dataset uses an inverted target encoding (`0` = disease present, `1` = healthy). The app corrects this automatically so predictions display correctly.
- Model training is cached with `@st.cache_resource` — retraining only happens on first load or after a code change.
