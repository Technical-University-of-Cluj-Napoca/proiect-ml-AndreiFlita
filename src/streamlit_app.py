import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Comparative Analysis", layout="wide")

st.sidebar.title("ML Project")
page = st.sidebar.radio("Navigation", [
    "Home",
    "Classification — Heart Disease",
    "Regression — Student Performance"
])

@st.cache_data
def load_cls_data():
    df = pd.read_csv('heart.csv')
    return df

@st.cache_data
def load_reg_data():
    df = pd.read_csv('Original_data_with_more_rows.csv')
    df = df.drop(columns=['Unnamed: 0'])
    return df

@st.cache_resource
def train_cls_models():
    df = load_cls_data()
    categorical_features = ['cp', 'restecg', 'slope']
    df_clean = df.drop(columns=['ca', 'thal'])
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)

    X = df_encoded.drop('target', axis=1)
    # In this dataset 0=disease, 1=healthy — invert so 1=disease (standard convention)
    y = 1 - df_encoded['target']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_cls_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
    scaler = StandardScaler()
    X_train = X_train_raw.copy().astype(float)
    X_test = X_test_raw.copy().astype(float)
    X_train[numeric_cls_cols] = scaler.fit_transform(X_train_raw[numeric_cls_cols])
    X_test[numeric_cls_cols] = scaler.transform(X_test_raw[numeric_cls_cols])

    models = {
        'Naïve Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
        'EBM': ExplainableBoostingClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
        }

    results_df = pd.DataFrame(results).T.sort_values('F1-Score', ascending=False)
    top5 = results_df.head(5).index.tolist()

    param_grids = {
        'Naïve Bayes': (GaussianNB(), {'var_smoothing': [1e-9, 1e-8, 1e-7]}),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1, 10]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5]}),
        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
        'SVM': (SVC(probability=True, random_state=42), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 11]}),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}),
        'CatBoost': (CatBoostClassifier(random_state=42, verbose=0), {'iterations': [50, 100], 'depth': [3, 5]}),
        'EBM': (ExplainableBoostingClassifier(random_state=42), {'max_bins': [128, 256]})
    }

    tuned = {}
    tuned_results = {}
    for name in top5:
        base_model, params = param_grids[name]
        gs = GridSearchCV(base_model, params, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        tuned[name] = best
        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        tuned_results[name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4),
            'Best Params': gs.best_params_
        }

    tuned_df = pd.DataFrame(tuned_results).T.sort_values('F1-Score', ascending=False)
    return models, tuned, results_df, tuned_df, X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

@st.cache_resource
def train_reg_models():
    df = load_reg_data()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=['MathScore'])
    y = df_encoded['MathScore']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test_sc = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(),
        'XGBoost': XGBRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
        'EBM': ExplainableBoostingRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        results[name] = {
            'MSE': round(mean_squared_error(y_test, y_pred), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'R2': round(r2_score(y_test, y_pred), 4)
        }

    results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
    top5 = results_df.head(5).index.tolist()

    param_grids = {
        'Linear Regression': (LinearRegression(), {'fit_intercept': [True, False]}),
        'Decision Tree': (DecisionTreeRegressor(random_state=42), {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5]}),
        'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}),
        'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}),
        'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 11]}),
        'XGBoost': (XGBRegressor(random_state=42), {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}),
        'CatBoost': (CatBoostRegressor(random_state=42, verbose=0), {'iterations': [50, 100], 'depth': [3, 5]}),
        'EBM': (ExplainableBoostingRegressor(random_state=42), {'max_bins': [128, 256]})
    }

    tuned = {}
    tuned_results = {}
    for name in top5:
        base_model, params = param_grids[name]
        gs = GridSearchCV(base_model, params, cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train_sc, y_train)
        best = gs.best_estimator_
        tuned[name] = best
        y_pred = best.predict(X_test_sc)
        tuned_results[name] = {
            'MSE': round(mean_squared_error(y_test, y_pred), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'R2': round(r2_score(y_test, y_pred), 4),
            'Best Params': gs.best_params_
        }

    tuned_df = pd.DataFrame(tuned_results).T.sort_values('R2', ascending=False)
    return models, tuned, results_df, tuned_df, X_train_sc, X_test_sc, y_train, y_test, scaler, X.columns.tolist()

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if page == "Home":
    st.title("Comparative Analysis of ML Models")
    st.markdown("""
    This project performs a comparative analysis of Machine Learning models for two supervised learning tasks:
    - **Classification** — Heart Disease prediction
    - **Regression** — Student Math Score prediction
    """)
    st.subheader("Technologies Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Libraries**\n- Pandas, NumPy\n- Scikit-learn\n- Matplotlib, Seaborn")
    with col2:
        st.markdown("**Models**\n- XGBoost, CatBoost\n- Random Forest, SVM\n- EBM, KNN")
    with col3:
        st.markdown("**Explainability**\n- SHAP Summary Plot\n- Waterfall & Force Plot\n- Scatter Plots")

# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────
elif page == "Classification — Heart Disease":
    st.title("Classification — Heart Disease Prediction")

    with st.spinner("Training models... this may take a minute."):
        models_cls, tuned_cls, results_df_cls, tuned_df_cls, X_train_cls, X_test_cls, y_train_cls, y_test_cls, scaler_cls, feature_names_cls = train_cls_models()

    df_cls = load_cls_data()

    with st.expander("About the Dataset"):
        st.write("""
        **Context:** This dataset contains clinical data about patients and the presence of heart disease.
        **Target:** 0 = No heart disease, 1 = Heart disease present
        **Features:** Age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and more.
        **Source:** UCI Heart Disease Dataset
        """)
        st.dataframe(df_cls.head())

    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df_cls['target'].value_counts().plot(kind='bar', color=['steelblue', 'coral'], ax=ax)
        ax.set_title('Heart Disease Distribution')
        ax.set_xlabel('Disease (0=No, 1=Yes)')
        ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(data=df_cls, x='age', hue='target', bins=30, ax=ax)
        ax.set_title('Age Distribution by Class')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_cls, x='target', y='chol', ax=ax)
        ax.set_title('Cholesterol vs Heart Disease')
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots()
        corr = df_cls.corr()['target'].drop('target').sort_values()
        corr.plot(kind='barh', ax=ax, color=['coral' if v < 0 else 'steelblue' for v in corr])
        ax.set_title('Feature Correlation with Target')
        st.pyplot(fig)

    st.subheader("Base Models — Comparison")
    st.dataframe(results_df_cls)

    st.subheader("Tuned Models — Comparison")
    st.dataframe(tuned_df_cls.drop(columns=['Best Params']))

    st.subheader("Model Selection & Prediction")
    selected_model_cls = st.selectbox("Select a model:", list(tuned_cls.keys()))

    st.markdown(f"**Best Hyperparameters:** `{tuned_df_cls.loc[selected_model_cls, 'Best Params']}`")

    st.markdown("**Model Metrics:**")
    m = tuned_df_cls.loc[selected_model_cls]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", m['Accuracy'])
    c2.metric("Precision", m['Precision'])
    c3.metric("Recall", m['Recall'])
    c4.metric("F1-Score", m['F1-Score'])
    c5.metric("ROC-AUC", m['ROC-AUC'])

    col_lc, col_cm = st.columns(2)
    with col_lc:
        st.markdown("**Learning Curve**")
        fig, ax = plt.subplots(figsize=(7, 5))
        model = tuned_cls[selected_model_cls]
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_cls, y_train_cls, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), scoring='f1'
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training')
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
        ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='orange', label='Validation')
        ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='orange')
        ax.set_title(f'Learning Curve — {selected_model_cls}')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        y_pred_sel = tuned_cls[selected_model_cls].predict(X_test_cls)
        cm = confusion_matrix(y_test_cls, y_pred_sel)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.subheader("Make a Prediction")

    default_age = 50; default_sex = 1; default_trestbps = 120; default_chol = 200
    default_fbs = 0; default_thalach = 150; default_exang = 0; default_oldpeak = 1.0
    default_cp = 0; default_restecg = 0; default_slope = 1

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 80, default_age)
        sex = st.selectbox("Sex", [0, 1], index=default_sex, format_func=lambda x: "Female" if x == 0 else "Male")
        trestbps = st.slider("Resting Blood Pressure", 80, 200, default_trestbps)
        chol = st.slider("Cholesterol", 100, 400, default_chol)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], index=default_fbs)
        thalach = st.slider("Max Heart Rate", 60, 220, default_thalach)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], index=default_exang)
        oldpeak = st.slider("ST Depression", 0.0, 6.0, float(default_oldpeak))
    with col3:
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=default_cp)
        restecg = st.selectbox("Resting ECG", [0, 1, 2], index=default_restecg)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2], index=default_slope)

    # Build input — zero out all features first then fill
    input_dict = {col: 0 for col in feature_names_cls}
    input_dict['age'] = age
    input_dict['sex'] = sex
    input_dict['trestbps'] = trestbps
    input_dict['chol'] = chol
    input_dict['fbs'] = fbs
    input_dict['thalach'] = thalach
    input_dict['exang'] = exang
    input_dict['oldpeak'] = oldpeak
    if cp == 1 and 'cp_1' in input_dict: input_dict['cp_1'] = 1
    elif cp == 2 and 'cp_2' in input_dict: input_dict['cp_2'] = 1
    elif cp == 3 and 'cp_3' in input_dict: input_dict['cp_3'] = 1
    if restecg == 1 and 'restecg_1' in input_dict: input_dict['restecg_1'] = 1
    elif restecg == 2 and 'restecg_2' in input_dict: input_dict['restecg_2'] = 1
    if slope == 1 and 'slope_1' in input_dict: input_dict['slope_1'] = 1
    elif slope == 2 and 'slope_2' in input_dict: input_dict['slope_2'] = 1

    input_df = pd.DataFrame([input_dict])[feature_names_cls]

    # Scale only numeric columns, leave one-hot as-is
    numeric_cls = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak']
    input_scaled = input_df.copy().astype(float)
    input_scaled[numeric_cls] = scaler_cls.transform(input_df[numeric_cls])

    if st.button("Predict", type="primary"):
        model = tuned_cls[selected_model_cls]
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"### Result: High Risk of Heart Disease (probability: {prob:.2%})")
        else:
            st.success(f"### Result: Low Risk of Heart Disease (probability: {prob:.2%})")

        st.subheader("SHAP Explanation for this Prediction")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_scaled)
            if isinstance(sv, list):
                sv_plot = sv[1]
                ev = explainer.expected_value[1]
            else:
                sv_plot = sv[:, :, 1] if len(sv.shape) == 3 else sv
                ev = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        except Exception:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train_cls, 50))
            sv_obj = explainer(input_scaled)
            sv_plot = sv_obj.values[:, :, 1]
            ev = explainer.expected_value[1]

        exp = shap.Explanation(
            values=sv_plot[0],
            base_values=ev,
            data=input_scaled.iloc[0].values,
            feature_names=feature_names_cls
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig)

# ─────────────────────────────────────────────
# REGRESSION
# ─────────────────────────────────────────────
elif page == "Regression — Student Performance":
    st.title("Regression — Student Math Score Prediction")

    with st.spinner("Training models... this may take a minute."):
        models_reg, tuned_reg, results_df_reg, tuned_df_reg, X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg, feature_names_reg = train_reg_models()

    df_reg = load_reg_data()

    with st.expander("About the Dataset"):
        st.write("""
        **Context:** This dataset contains information about students and their exam performance.
        **Target:** MathScore — predicted from ReadingScore, WritingScore and demographic features.
        **Features:** Gender, Ethnic Group, Parent Education, Lunch Type, Test Preparation, Reading & Writing Scores.
        **Source:** Students Exam Scores Dataset — Kaggle
        """)
        st.dataframe(df_reg.head())

    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df_reg, x='MathScore', bins=30, color='steelblue', ax=ax)
        ax.set_title('Math Score Distribution')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_reg, x='Gender', y='MathScore', ax=ax)
        ax.set_title('Math Score by Gender')
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        sns.boxplot(data=df_reg, x='TestPrep', y='MathScore', ax=ax)
        ax.set_title('Math Score by Test Preparation')
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots()
        parent_order = df_reg.groupby('ParentEduc')['MathScore'].mean().sort_values().index
        sns.boxplot(data=df_reg, x='ParentEduc', y='MathScore', order=parent_order, ax=ax)
        ax.set_title('Math Score by Parent Education')
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_reg, x='ReadingScore', y='MathScore', alpha=0.3, ax=ax)
        ax.set_title('Reading vs Math Score')
        st.pyplot(fig)
    with col6:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_reg, x='WritingScore', y='MathScore', alpha=0.3, ax=ax)
        ax.set_title('Writing vs Math Score')
        st.pyplot(fig)

    st.subheader("Base Models — Comparison")
    st.dataframe(results_df_reg)

    st.subheader("Tuned Models — Comparison")
    st.dataframe(tuned_df_reg.drop(columns=['Best Params']))

    st.subheader("Model Selection & Prediction")
    selected_model_reg = st.selectbox("Select a model:", list(tuned_reg.keys()))

    st.markdown(f"**Best Hyperparameters:** `{tuned_df_reg.loc[selected_model_reg, 'Best Params']}`")

    st.markdown("**Model Metrics:**")
    m = tuned_df_reg.loc[selected_model_reg]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", m['MSE'])
    c2.metric("MAE", m['MAE'])
    c3.metric("RMSE", m['RMSE'])
    c4.metric("R²", m['R2'])

    col_lc, col_res = st.columns(2)
    with col_lc:
        st.markdown("**Learning Curve**")
        fig, ax = plt.subplots(figsize=(7, 5))
        model = tuned_reg[selected_model_reg]
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_reg, y_train_reg, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), scoring='r2'
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='blue', label='Training')
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
        ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='orange', label='Validation')
        ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='orange')
        ax.set_title(f'Learning Curve — {selected_model_reg}')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('R² Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_res:
        st.markdown("**Predictions vs Actual**")
        fig, ax = plt.subplots(figsize=(7, 5))
        y_pred_sel = tuned_reg[selected_model_reg].predict(X_test_reg)
        ax.scatter(y_test_reg, y_pred_sel, alpha=0.3, color='steelblue')
        ax.plot([y_test_reg.min(), y_test_reg.max()],
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{selected_model_reg} — Predictions vs Actual')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.subheader("Make a Prediction")

    d_gender = "male"; d_ethnic = "group C"; d_parent = "some college"
    d_lunch = "standard"; d_prep = "none"; d_read = 65; d_write = 65

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["male", "female"], index=["male", "female"].index(d_gender))
        ethnic_group = st.selectbox("Ethnic Group", ["group A", "group B", "group C", "group D", "group E"],
                                    index=["group A", "group B", "group C", "group D", "group E"].index(d_ethnic))
        parent_educ = st.selectbox("Parent Education",
                                   ["some high school", "high school", "some college",
                                    "associate's degree", "bachelor's degree", "master's degree"],
                                   index=["some high school", "high school", "some college",
                                          "associate's degree", "bachelor's degree", "master's degree"].index(d_parent))
        lunch_type = st.selectbox("Lunch Type", ["standard", "free/reduced"],
                                  index=["standard", "free/reduced"].index(d_lunch))
        test_prep = st.selectbox("Test Preparation", ["none", "completed"],
                                 index=["none", "completed"].index(d_prep))
    with col2:
        reading_score = st.slider("Reading Score", 0, 100, d_read)
        writing_score = st.slider("Writing Score", 0, 100, d_write)

    input_dict = {col: 0 for col in feature_names_reg}
    input_dict['ReadingScore'] = reading_score
    input_dict['WritingScore'] = writing_score
    if 'Gender_male' in input_dict: input_dict['Gender_male'] = 1 if gender == 'male' else 0
    if 'EthnicGroup_group B' in input_dict: input_dict['EthnicGroup_group B'] = 1 if ethnic_group == 'group B' else 0
    if 'EthnicGroup_group C' in input_dict: input_dict['EthnicGroup_group C'] = 1 if ethnic_group == 'group C' else 0
    if 'EthnicGroup_group D' in input_dict: input_dict['EthnicGroup_group D'] = 1 if ethnic_group == 'group D' else 0
    if 'EthnicGroup_group E' in input_dict: input_dict['EthnicGroup_group E'] = 1 if ethnic_group == 'group E' else 0
    if "ParentEduc_bachelor's degree" in input_dict: input_dict["ParentEduc_bachelor's degree"] = 1 if parent_educ == "bachelor's degree" else 0
    if 'ParentEduc_high school' in input_dict: input_dict['ParentEduc_high school'] = 1 if parent_educ == 'high school' else 0
    if "ParentEduc_master's degree" in input_dict: input_dict["ParentEduc_master's degree"] = 1 if parent_educ == "master's degree" else 0
    if 'ParentEduc_some college' in input_dict: input_dict['ParentEduc_some college'] = 1 if parent_educ == 'some college' else 0
    if 'ParentEduc_some high school' in input_dict: input_dict['ParentEduc_some high school'] = 1 if parent_educ == 'some high school' else 0
    if 'LunchType_standard' in input_dict: input_dict['LunchType_standard'] = 1 if lunch_type == 'standard' else 0
    if 'TestPrep_none' in input_dict: input_dict['TestPrep_none'] = 1 if test_prep == 'none' else 0

    input_df = pd.DataFrame([input_dict])[feature_names_reg]

    input_scaled = pd.DataFrame(scaler_reg.transform(input_df), columns=feature_names_reg)

    if st.button("Predict Math Score", type="primary"):
        model = tuned_reg[selected_model_reg]
        pred = model.predict(input_scaled)[0]
        st.success(f"### Predicted Math Score: {pred:.1f} / 100")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(['Predicted', 'Maximum'], [pred, 100], color=['steelblue', 'lightgray'])
        ax.set_xlim(0, 110)
        ax.set_title('Predicted vs Maximum Score')
        for i, v in enumerate([pred, 100]):
            ax.text(v + 1, i, f'{v:.1f}', va='center')
        st.pyplot(fig)

        st.subheader("SHAP Explanation for this Prediction")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_scaled)
            ev = explainer.expected_value
        except Exception:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train_reg, 50))
            sv = explainer.shap_values(input_scaled)
            ev = explainer.expected_value

        exp = shap.Explanation(
            values=sv[0],
            base_values=ev,
            data=input_scaled.iloc[0].values,
            feature_names=feature_names_reg
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig)