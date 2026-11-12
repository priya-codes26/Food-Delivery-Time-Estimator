from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "food_delivery.csv"
MODEL_PATH = BASE_DIR / "model" / "food_delivery_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

st.set_page_config(
    page_title="Food Delivery Time Estimator",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple styling for a clean project dashboard.
st.markdown("""
<style>
    .main {background-color: #f7f8fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e8eaf2;
    }
    .hero {
        padding: 22px; border-radius: 16px; color: white;
        background: linear-gradient(110deg, #172554, #4338ca);
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

with st.sidebar:
    st.title("🍔 Delivery Estimator")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "⏱️ Predict Delivery", "📊 Analytics", "ℹ️ About"]
    )
    st.caption("Machine Learning Major Project")

st.markdown("""
<div class="hero">
    <h1>Food Delivery Time Estimator</h1>
    <p>Estimate food delivery time using machine learning.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Trained model not found. First run: python train_model.py")
    st.info("After training, the model will be saved inside the model folder.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

if page == "🏠 Overview":
    st.subheader("Project Overview")
    st.write(
        "This dashboard predicts the expected delivery time based on distance, "
        "weather, traffic, vehicle type, food preparation time and courier experience."
    )

    total_predictions = len(st.session_state.history)
    avg_delivery = (
        sum(item["Predicted_Time_min"] for item in st.session_state.history)
        / total_predictions if total_predictions else 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset records", f"{len(data):,}" if not data.empty else "N/A")
    c2.metric("Predictions made", total_predictions)
    c3.metric("Average predicted time", f"{avg_delivery:.1f} min" if total_predictions else "—")

    st.markdown("### How it works")
    st.write("1. Enter delivery details.  2. The trained ML model processes the inputs.")
    st.write("3. The dashboard displays the estimated delivery time.")

    if not data.empty:
        fig = px.histogram(
            data, x="Delivery_Time_min", nbins=25,
            title="Delivery Time Distribution in the Dataset",
            labels={"Delivery_Time_min": "Delivery time (minutes)"},
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "⏱️ Predict Delivery":
    st.subheader("Enter delivery details")
    st.caption("Change the values and click Predict to get an estimate.")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            distance = st.slider("Delivery distance (km)", 1.0, 20.0, 5.0, 0.5)
            weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
            traffic = st.selectbox("Traffic level", ["Low", "Medium", "High"])

        with right:
            vehicle = st.selectbox("Vehicle type", ["Bike", "Scooter", "Car"])
            preparation = st.slider("Food preparation time (minutes)", 5, 30, 15)
            experience = st.slider("Courier experience (years)", 0.0, 10.0, 2.0, 0.5)

        submitted = st.form_submit_button("🚀 Predict Delivery Time", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": preparation,
            "Courier_Experience_yrs": experience,
        }])

        predicted_time = float(model.predict(input_data)[0])
        predicted_time = max(1.0, predicted_time)

        st.success("Prediction completed!")
        st.metric("Estimated delivery time", f"{predicted_time:.1f} minutes")
        st.info(f"Approximately {int(round(predicted_time))} minutes. This is a model estimate, not a guaranteed arrival time.")

        st.session_state.history.append({
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": preparation,
            "Courier_Experience_yrs": experience,
            "Predicted_Time_min": round(predicted_time, 1),
        })

    if st.session_state.history:
        st.markdown("### Recent predictions")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download prediction history (CSV)",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()

elif page == "📊 Analytics":
    st.subheader("Dataset Analytics")
    if data.empty:
        st.warning("Dataset file is not available.")
    else:
        a, b = st.columns(2)
        with a:
            fig1 = px.scatter(
                data, x="Distance_km", y="Delivery_Time_min",
                color="Traffic_Level", title="Distance vs Delivery Time",
                labels={"Distance_km": "Distance (km)", "Delivery_Time_min": "Delivery time (min)"}
            )
            st.plotly_chart(fig1, use_container_width=True)
        with b:
            fig2 = px.box(
                data, x="Weather", y="Delivery_Time_min",
                color="Weather", title="Delivery Time by Weather",
                labels={"Delivery_Time_min": "Delivery time (min)"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            data.groupby("Traffic_Level", as_index=False)["Delivery_Time_min"].mean(),
            x="Traffic_Level", y="Delivery_Time_min",
            title="Average Delivery Time by Traffic Level",
            labels={"Delivery_Time_min": "Average delivery time (min)"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.markdown("### Model performance on test data")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f'{metrics["mae_minutes"]} min')
        m2.metric("RMSE", f'{metrics["rmse_minutes"]} min')
        m3.metric("R² score", f'{metrics["r2_score"]}')

elif page == "ℹ️ About":
    st.subheader("About this project")
    st.write(
        "The Food Delivery Time Estimator is a machine learning application "
        "that estimates delivery duration from order and delivery conditions."
    )
    st.markdown("""
    **Algorithm:** Random Forest Regressor

    **Input features:** Distance, weather, traffic, vehicle type,
    preparation time and courier experience.

    **Target:** Delivery time in minutes.

    **Libraries:** Python, Pandas, Scikit-learn, Streamlit and Plotly.
    """)
    st.warning(
        "The included starter dataset is synthetic (computer-generated) for "
        "learning and demonstration. Replace it with a properly sourced real-world "
        "dataset and retrain the model before making real-world accuracy claims."
    )

st.divider()
st.caption("Food Delivery Time Estimator | Machine Learning Project")
