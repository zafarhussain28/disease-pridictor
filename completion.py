import streamlit as st
st.markdown("""
            <style>
            .stApp { set-background-color: white; }
            </style>
            """, unsafe_allow_html=True)

st.title("HEALTH AI PRIDICTOR")
st.markdown(
    "<h1 style='color: black; text-align: center; font-weight: bold;'>HEALTH AI PRIDICTOR</h1>",
    unsafe_allow_html=True
)
# Remove the default padding/margins to make the image edge-to-edge


# Add an edge-to-edge image
st.image(
    "https://unsplash.com/photos/Cy_RRgdwHxA/download?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&force=true&w=640",
   
    caption="know your health better with AI"
)
st.markdown("""
            <style> 
            .stApp { 
                border-radius: 40px;
            }
            </style>
            
            """, unsafe_allow_html=True)
c = st.text_input(
    "Enter your name",
    placeholder="Enter your name here",
    key="name_input"
)

# Custom CSS for input border and placeholder color
st.markdown("""
    <style>
    /* Change border color */
    input[data-testid="stTextInput"][id^="name_input"] {
        border: 2px solid #4CAF50 !important;  /* Green border */
        border-radius: 8px !important;
    }
    /* Change placeholder color */
    input[data-testid="stTextInput"][id^="name_input"]::placeholder {
        color: #87CEFA !important;  /* Light blue placeholder */
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
d = st.number_input("Enter your age", placeholder="Age",min_value= 1, max_value=100, step=1, key="age_input")
st.write("Your name is:", c)
st.write("Your age is:", d)
st.success("Your details have been recorded successfully!")
st.warning(" This is an AI-based health predictor INFORMATION MAY NOT ALWAYS BE ACCURATE")
# new code
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Predictor App")
st.write(
    "Enter your symptoms below and click **Predict Disease** to see the likely diagnosis along with home remedies and when to consult a doctor."
)

# ✅ Load and clean dataset
df = pd.read_csv("Symptom2Disease.csv")

df = df.drop(columns=["Unnamed: 0"], errors='ignore')
df.columns = df.columns.str.strip()
df = df.dropna()

# ✅ Features and labels
X_raw = df['text']
y_raw = df['label']

# ✅ Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# ✅ TF-IDF Vectorizer on symptoms text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(X_raw)

# ✅ Train model (no need to split again for app)
model = LogisticRegression(max_iter=500, solver='saga')
model.fit(X, y_encoded)

# ✅ Remedies database
remedy_database = {
    "migraine": {
        "description": "Migraine is a neurological condition with recurring headaches, often accompanied by nausea and light sensitivity.",
        "remedy": "Apply a cold compress, rest in a dark room, and avoid known triggers like caffeine or noise.",
        "consult": "Consult a neurologist if migraines are frequent or disabling."
    },
    "common cold": {
        "description": "Common cold is a viral respiratory infection causing congestion, runny nose, sneezing, and sore throat.",
        "remedy": "Stay hydrated, use steam inhalation, and drink warm fluids like ginger tea.",
        "consult": "See a doctor if fever lasts more than 3 days or breathing becomes difficult."
    },
    # ➔ Add more diseases as needed from your original list...
}

# ✅ Normalize keys
remedy_database = {k.lower(): v for k, v in remedy_database.items()}

# ✅ User input section
user_input = st.text_area("📝 Enter your symptoms here:")

if st.button("🔍 Predict Disease"):
    if not user_input.strip():
        st.warning("Please enter your symptoms before predicting.")
    else:
        input_vector = vectorizer.transform([user_input])
        pred_encoded = model.predict(input_vector)[0]
        pred_disease = le.inverse_transform([pred_encoded])[0]

        st.success(f"### 🦠 Predicted Disease: **{pred_disease}**")

        key = pred_disease.lower()
        info = remedy_database.get(key)

        if info:
            st.write(f"**🩺 Description:**\n{info['description']}")
            st.write(f"**🌿 Home Remedies:**\n{info['remedy']}")
            st.write(f"**⚠ When to Consult:**\n{info['consult']}")
        else:
            st.info("No additional info available for this disease yet.")
            
