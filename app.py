"""Optional Streamlit dashboard for an educational demand forecasting run."""
import streamlit as st

from src.pipeline import run_forecast


st.set_page_config(page_title="PowerWise Forecast Lab", layout="wide")
st.title("PowerWise · Electricity Forecast Lab")
st.caption("A learning demonstration using a synthetic time series. It is not utility data or an operational forecast.")

with st.sidebar.form("run_settings"):
    st.header("Training settings")
    epochs = st.slider("Epochs", 25, 600, 300, 25)
    seed = st.number_input("Random seed", min_value=0, value=7)
    submitted = st.form_submit_button("Run forecast")

if submitted:
    with st.spinner("Training the NumPy network and evaluating the chronological test period..."):
        metrics = run_forecast(epochs=epochs, seed=int(seed))
    st.subheader("Held-out test metrics")
    st.json(metrics)
    c1, c2 = st.columns(2)
    c1.image("reports/latest/predictions.png", caption="Actual, MLP, and persistence predictions")
    c2.image("reports/latest/loss.png", caption="Training and validation loss")
    st.image("reports/latest/errors.png", caption="Test-set residuals")
else:
    st.info("Choose training settings and press Run forecast. The default data is synthetic and clearly labeled.")
