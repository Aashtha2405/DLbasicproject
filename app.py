import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import plotly.express as px
st.set_page_config(
    page_title="PlantGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(46,125,50,0.18), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(129,199,132,0.12), transparent 30%),
        linear-gradient(135deg, #061a12 0%, #0b2b1d 50%, #05140d 100%);
    color: white;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* Header */
.hero {
    text-align: center;
    padding: 35px 20px 25px;
}

.logo {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 4px;
    color: #8df5a7;
}

.hero h1 {
    font-size: 58px;
    margin: 8px 0;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #8df5a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #b7cfc1;
    font-size: 18px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(14px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}

.small-text {
    color: #a9c1b4;
}

/* Result */
.result-card {
    background: linear-gradient(
        135deg,
        rgba(35,150,75,0.25),
        rgba(255,255,255,0.06)
    );
    border: 1px solid rgba(111,255,145,0.25);
    border-radius: 24px;
    padding: 30px;
    margin-top: 20px;
}

.result-label {
    color: #91e6a8;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-title {
    font-size: 34px;
    font-weight: 800;
    margin-top: 5px;
}

/* Metric */
.metric-box {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #8df5a7;
}

.metric-label {
    color: #a9c1b4;
    font-size: 14px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    padding: 13px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #39b56a, #7be495);
    color: #062014;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(92,220,130,0.3);
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border-radius: 18px;
    padding: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #779386;
    padding: 35px 0 10px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_cnn.keras"
    )

    label_encoder = joblib.load(
        "plant_disease_label_encoder.pkl"
    )

    return model, label_encoder

try:

    model, label_encoder = load_model()
    model_loaded = True

except Exception as e:

    model_loaded = False
    model = None
    label_encoder = None


st.markdown("""
<div class="hero">

<div class="logo">🌿 PLANTGUARD AI</div>

<h1>Plant Disease Detection</h1>

<p>
AI-powered plant health analysis using Deep Learning
</p>

</div>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="metric-box">

    <div class="metric-value">🤖 CNN</div>

    <div class="metric-label">
    Deep Learning Model
    </div>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="metric-box">

    <div class="metric-value">🌱 4+</div>

    <div class="metric-label">
    Plant Categories
    </div>

    </div>
    """, unsafe_allow_html=True)
with col3:

    st.markdown("""
    <div class="metric-box">

    <div class="metric-value">⚡ AI</div>

    <div class="metric-label">
    Instant Prediction
    </div>

    </div>
    """, unsafe_allow_html=True)


st.write("")


left, right = st.columns([1, 1], gap="large")


with left:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    📸 Upload Leaf Image
    </div>

    <p class="small-text">
    Upload a clear image of the plant leaf.
    Our CNN model will analyze the image.
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )


with right:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    🔬 AI Analysis
    </div>

    <p class="small-text">
    The uploaded image will be resized and
    processed before being passed to the CNN.
    </p>

    <br>

    <b>Supported plants</b>

    <p>
    🍎 Apple &nbsp;&nbsp;
    🌽 Corn &nbsp;&nbsp;
    🥔 Potato &nbsp;&nbsp;
    🍅 Tomato
    </p>

    </div>
    """, unsafe_allow_html=True)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("")

    image_col, prediction_col = st.columns([1, 1], gap="large")


    with image_col:

        st.markdown(
            '<div class="card-title">🖼️ Leaf Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    with prediction_col:

        st.markdown(
            '<div class="card-title">🧠 AI Diagnosis</div>',
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🔍 ANALYZE LEAF"
        )


        if analyze:

            if not model_loaded:

                st.error(
                    "Model files are not available yet. "
                    "Train and save your CNN model first."
                )

            else:

                with st.spinner(
                    "🧠 CNN is analyzing the leaf..."
                ):

                    img = image.resize((128, 128))

                    img_array = np.array(img)

                    img_array = img_array / 255.0

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )
                    prediction = model.predict(
                        img_array,
                        verbose=0
                    )

                    predicted_index = np.argmax(
                        prediction[0]
                    )

                    confidence = (
                        prediction[0][predicted_index]
                        * 100
                    )

                    predicted_class = (
                        label_encoder.inverse_transform(
                            [predicted_index]
                        )[0]
                    )

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-label">
                    DETECTED CONDITION
                    </div>

                    <div class="result-title">
                    {predicted_class}
                    </div>

                    <br>

                    <div class="result-label">
                    MODEL CONFIDENCEcd "CNN project"
                    </div>

                    <div class="result-title">
                    {confidence:.2f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

if uploaded_file is not None and model_loaded:

    if "prediction" in locals():

        probabilities = prediction[0] * 100

        class_names = label_encoder.classes_

        result_df = {
            "Disease": class_names,
            "Probability": probabilities
        }

        import pandas as pd

        result_df = pd.DataFrame(result_df)

        result_df = result_df.sort_values(
            "Probability",
            ascending=False
        ).head(8)

        st.write("")

        st.markdown(
            '<div class="card-title">📊 Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        fig = px.bar(
            result_df,
            x="Probability",
            y="Disease",
            orientation="h",
            text="Probability"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Confidence (%)",
            yaxis_title="",
            margin=dict(l=10, r=20, t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.write("")

st.markdown("""
<div class="card">

<div class="card-title">
⚙️ How PlantGuard AI Works
</div>

<p class="small-text">

<b>01 — Upload</b><br>
Upload a clear image of a plant leaf.

<br><br>

<b>02 — Preprocessing</b><br>
The image is resized and pixel values are normalized.

<br><br>

<b>03 — CNN Analysis</b><br>
The Convolutional Neural Network automatically extracts
important visual patterns from the leaf.

<br><br>

<b>04 — Classification</b><br>
The final Softmax layer calculates probabilities for
the different disease classes.

<br><br>

<b>05 — Result</b><br>
The class with the highest probability is displayed
as the predicted condition.

</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">

🌿 PlantGuard AI • Plant Disease Detection using Deep Learning

<br>

Built with Python • TensorFlow • Keras • Streamlit

</div>
""", unsafe_allow_html=True)