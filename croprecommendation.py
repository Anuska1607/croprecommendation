import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load saved models
@st.cache_resource
def load_models():
    try:
        rf_class = joblib.load('crop_classifier.joblib')
        rf_reg = joblib.load('env_regressor.joblib')
        le = joblib.load('label_encoder.joblib')
        return rf_class, rf_reg, le
    except:
        st.error("Model files not found. Please run train_models.py first")
        st.stop()

rf_class, rf_reg, le = load_models()

# Load sample data for visualization
@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('crop_recommendation.csv')
    except:
        st.warning("Using dummy data for visualization as dataset file wasn't found.")
        np.random.seed(42)
        N = 2200
        return pd.DataFrame({
            'N': np.random.randint(0, 140, N),
            'P': np.random.randint(5, 145, N),
            'K': np.random.randint(5, 205, N),
            'ph': np.random.uniform(3.5, 10, N),
            'rainfall': np.random.uniform(20, 300, N),
            'label': np.random.choice(le.classes_, N)
        })

df = load_sample_data()

# Streamlit UI
st.title('🌱 Smart Crop Recommendation System')
st.write("""
This app recommends the most suitable crop based on soil conditions (N, P, K, pH) and rainfall,
and predicts expected temperature and humidity using pre-trained Random Forest models.
""")

# Sidebar with info
st.sidebar.header('About')
st.sidebar.info("""
- **N (Nitrogen)**: 0-140 kg/ha
- **P (Phosphorous)**: 5-145 kg/ha
- **K (Potassium)**: 5-205 kg/ha
- **pH**: 3.5-10
- **Rainfall**: 20-300 mm
""")

# Input form
st.header('Input Parameters')
col1, col2 = st.columns(2)

with col1:
    N = st.slider('Nitrogen (N) kg/ha', 0, 140, 50)
    P = st.slider('Phosphorous (P) kg/ha', 5, 145, 50)
    K = st.slider('Potassium (K) kg/ha', 5, 205, 50)

with col2:
    ph = st.slider('pH Value', 3.5, 10.0, 6.5)
    rainfall = st.slider('Rainfall (mm)', 20.0, 300.0, 100.0)

# Make predictions
input_data = [[N, P, K, ph, rainfall]]
crop_pred = rf_class.predict(input_data)[0]
crop_name = le.inverse_transform([crop_pred])[0]
temp_humidity_pred = rf_reg.predict(input_data)[0]

# Display results
st.header('Recommendation Results')
st.subheader(f'Recommended Crop: **{crop_name}**')

# Show probabilities
probs = rf_class.predict_proba(input_data)[0]
top_n = 5
top_indices = probs.argsort()[-top_n:][::-1]
top_probs = probs[top_indices]
top_crops = le.inverse_transform(top_indices)

st.subheader('Top 5 Crop Probabilities')
fig, ax = plt.subplots()
sns.barplot(x=top_probs, y=top_crops, palette='viridis', ax=ax)
ax.set_xlabel('Probability')
ax.set_ylabel('Crop')
st.pyplot(fig)

# Show regression results
st.subheader('Expected Environmental Conditions')
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Temperature (°C)", value=f"{temp_humidity_pred[0]:.1f}")
with col2:
    st.metric(label="Predicted Humidity (%)", value=f"{temp_humidity_pred[1]:.1f}")

# Data visualization
st.header('Data Exploration')
selected_feature = st.selectbox('Select feature to visualize', ['N', 'P', 'K', 'ph', 'rainfall'])
selected_crop = st.selectbox('Select crop to filter', ['All'] + list(le.classes_))

if selected_crop == 'All':
    filtered_df = df
else:
    filtered_df = df[df['label'] == selected_crop]

fig2, ax2 = plt.subplots()
sns.histplot(filtered_df[selected_feature], kde=True, ax=ax2)
ax2.set_title(f'Distribution of {selected_feature} for {selected_crop}')
st.pyplot(fig2)