import time
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# ===== General Configuration =====
DATA_PATH = "customer_churn_dataset-training-master.csv"
TARGET_COL = "Churn"
DROP_COLS = ["CustomerID"]

CATEGORICAL_COLS = ["Gender", "Subscription Type", "Contract Length"]


# ===== Load and Prepare Data =====
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # drop rows with missing values
    df = df.dropna().copy()

    return df


def preprocess(df):
    df = df.copy()

    # Target column
    y = df[TARGET_COL].astype(int)

    # drop ID and Target
    X = df.drop(columns=[TARGET_COL] + DROP_COLS)

    # one-hot encode categorical features
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    return X, y


# ===== Train 5 Models =====
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scaler for models that require scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "use_scaled": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=8, random_state=42),
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=150, max_depth=10, random_state=42
            ),
            "use_scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "use_scaled": False,
        },
        # simple Neural Network / Deep Learning model
        "Neural Network (MLP)": {
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=42,
            ),
            "use_scaled": True,
        },
    }

    results = []
    trained_models = {}

    for name, cfg in models.items():
        model = cfg["model"]
        use_scaled = cfg["use_scaled"]

        if use_scaled:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        start_train = time.time()
        model.fit(X_tr, y_train)
        end_train = time.time()

        start_pred = time.time()
        y_pred = model.predict(X_te)
        end_pred = time.time()

        acc = accuracy_score(y_test, y_pred)
        train_time = end_train - start_train
        pred_time = end_pred - start_pred

        # approximate complexity
        if "Logistic" in name:
            complexity = "Low"
        elif "Decision" in name:
            complexity = "Medium"
        elif "Random" in name or "Gradient" in name:
            complexity = "High"
        elif "Neural" in name:
            complexity = "Very High"
        else:
            complexity = "Unknown"

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Train Time (s)": round(train_time, 4),
                "Predict Time (s)": round(pred_time, 6),
                "Complexity": complexity,
            }
        )

        trained_models[name] = {
            "model": model,
            "use_scaled": use_scaled,
            "scaler": scaler if use_scaled else None,
            "X_columns": X.columns.tolist(),
        }

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["Model"]

    return results_df, trained_models, best_model_name


# ===== Input Form for a Single Customer =====
def build_input_form(df_raw):
    st.subheader("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            int(df_raw["Age"].min()),
            int(df_raw["Age"].max()),
            int(df_raw["Age"].median()),
        )
        tenure = st.number_input(
            "Tenure (months)",
            int(df_raw["Tenure"].min()),
            int(df_raw["Tenure"].max()),
            int(df_raw["Tenure"].median()),
        )
        usage = st.number_input(
            "Usage Frequency",
            int(df_raw["Usage Frequency"].min()),
            int(df_raw["Usage Frequency"].max()),
            int(df_raw["Usage Frequency"].median()),
        )
        support_calls = st.number_input(
            "Support Calls",
            int(df_raw["Support Calls"].min()),
            int(df_raw["Support Calls"].max()),
            int(df_raw["Support Calls"].median()),
        )
        pay_delay = st.number_input(
            "Payment Delay",
            int(df_raw["Payment Delay"].min()),
            int(df_raw["Payment Delay"].max()),
            int(df_raw["Payment Delay"].median()),
        )

    with col2:
        total_spend = st.number_input(
            "Total Spend",
            float(df_raw["Total Spend"].min()),
            float(df_raw["Total Spend"].max()),
            float(df_raw["Total Spend"].median()),
        )
        last_interaction = st.number_input(
            "Last Interaction",
            int(df_raw["Last Interaction"].min()),
            int(df_raw["Last Interaction"].max()),
            int(df_raw["Last Interaction"].median()),
        )

        gender = st.selectbox("Gender", sorted(df_raw["Gender"].dropna().unique()))
        sub_type = st.selectbox(
            "Subscription Type",
            sorted(df_raw["Subscription Type"].dropna().unique()),
        )
        contract = st.selectbox(
            "Contract Length",
            sorted(df_raw["Contract Length"].dropna().unique()),
        )

    input_dict = {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support_calls,
        "Payment Delay": pay_delay,
        "Total Spend": total_spend,
        "Last Interaction": last_interaction,
        "Gender": gender,
        "Subscription Type": sub_type,
        "Contract Length": contract,
    }

    return pd.DataFrame([input_dict])


def preprocess_single_input(input_df, df_raw):
    # ensure same one-hot encoding as full dataset
    temp = pd.concat(
        [input_df, df_raw.drop(columns=[TARGET_COL] + DROP_COLS)], axis=0
    )

    temp = pd.get_dummies(temp, columns=CATEGORICAL_COLS, drop_first=True)

    input_processed = temp.iloc[[0]]

    return input_processed


def predict_single(model_info, X_single):
    model = model_info["model"]
    scaler = model_info["scaler"]
    use_scaled = model_info["use_scaled"]

    if use_scaled:
        X_single = scaler.transform(X_single)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X_single)[0, 1])
    else:
        prob = None

    pred = int(model.predict(X_single)[0])
    return pred, prob


# ===== Streamlit App =====
def main():
    st.set_page_config(page_title="Customer Churn – Streamlit App", layout="wide")

    st.title("Customer Churn Prediction – Machine Learning Project")
    st.write(
        """
        This application is a **Real Use Case Deployment** for the Customer Churn project:

        - Uses **5 models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Neural Network (MLP).
        - Compares models based on **Accuracy**, **Time**, and **Complexity**.
        - Uses the **best model** to predict churn for a new customer.
        """
    )

    df_raw = load_data()
    X, y = preprocess(df_raw)
    results_df, trained_models, best_model_name = train_models(X, y)

    tab1, tab2 = st.tabs(["🔍 Model Comparison", "📊 Single Customer Prediction"])

    # --- Tab 1: Model Comparison ---
    with tab1:
        st.subheader("Models Performance Comparison")
        st.dataframe(results_df, use_container_width=True)

        st.markdown(f"### ✅ Best Model by Accuracy: **{best_model_name}**")

        st.bar_chart(results_df.set_index("Model")["Accuracy"])

        st.caption("Complexity is an approximate estimation based on model type.")

    # --- Tab 2: Single Customer Prediction ---
    with tab2:
        st.subheader("Predict Churn for a Single Customer")

        model_name = st.selectbox(
            "Choose a model:",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name),
        )

        model_info = trained_models[model_name]

        input_df = build_input_form(df_raw)

        if st.button("Predict Churn"):
            X_single = preprocess_single_input(input_df, df_raw)
            pred, prob = predict_single(model_info, X_single)

            if pred == 1:
                if prob is not None:
                    st.error(
                        f"Result: Customer **WILL Churn** (probability ≈ {prob:.3f})."
                    )
                else:
                    st.error("Result: Customer **WILL Churn**.")
            else:
                if prob is not None:
                    st.success(
                        f"Result: Customer **WILL NOT Churn** (probability ≈ {prob:.3f})."
                    )
                else:
                    st.success("Result: Customer **WILL NOT Churn**.")

            st.write("### Customer Input Data")
            st.write(input_df)


if __name__ == "__main__":
    main()
