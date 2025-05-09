import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessing objects
model = load_model("score_predictor_model.h5")
scaler = joblib.load("scaler.pkl")
venue_encoder = joblib.load("venue_encoder.pkl")
batting_team_encoder = joblib.load("batting_team_encoder.pkl")
bowling_team_encoder = joblib.load("bowling_team_encoder.pkl")
striker_encoder = joblib.load("striker_encoder.pkl")
bowler_encoder = joblib.load("bowler_encoder.pkl")

# Load data for dropdowns
df = pd.read_csv("ipl_data.csv")

st.title("🏏 IPL Score Predictor")

venue = st.selectbox("Select Venue", sorted(df["venue"].unique()))
batting_team = st.selectbox("Select Batting Team", sorted(df["bat_team"].unique()))
bowling_team = st.selectbox("Select Bowling Team", sorted(df["bowl_team"].unique()))
batsman = st.selectbox("Select Striker", sorted(df["batsman"].unique()))
bowler = st.selectbox("Select Bowler", sorted(df["bowler"].unique()))

if st.button("Predict Score"):
    try:
        # Encode
        v = venue_encoder.transform([venue])[0]
        bt = batting_team_encoder.transform([batting_team])[0]
        blt = bowling_team_encoder.transform([bowling_team])[0]
        s = striker_encoder.transform([batsman])[0]
        b = bowler_encoder.transform([bowler])[0]

        # Make input array
        input_data = np.array([[v, bt, blt, s, b]])
        input_scaled = scaler.transform(input_data)

        # Predict
        predicted_score = model.predict(input_scaled)
        predicted_score = int(predicted_score[0][0])

        st.success(f"🏆 Predicted Score: {predicted_score}")
    except Exception as e:
        st.error(f"Error: {e}")
