import streamlit as st
import pandas as pd
from PIL import Image
from utils.inference import detect_disease, get_all_disease_classes

# ─── PAGE CONFIGURATION ────────────────────────────────────────────────────
st.set_page_config(page_title="AI for Farmers", layout="wide")

# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────
@st.cache_data
def load_disease_catalog():
    df = pd.read_csv("utils/data/disease_catalog.csv")
    df["Disease Key"] = df.iloc[:, 0] + " " + df.iloc[:, 1]  # Combine plant_type + disease
    return df

def load_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom styling
load_css("style.css")

# ─── HERO SECTION ──────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>AI for Farmers: Plant Disease Diagnosis</h1>
    <p>Detect plant diseases and receive AI-generated treatment recommendations</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT: Upload | Detection | Advice ──────────────────────────────
col1, col2, col3 = st.columns(3)

# 📤 IMAGE UPLOAD
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state["uploaded_image"] = image
    st.markdown("</div>", unsafe_allow_html=True)

# 🔍 DISEASE DETECTION
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Detection Result")

    if "uploaded_image" in st.session_state:
        if st.button("Run Detection"):
            result = detect_disease(st.session_state["uploaded_image"])
            st.session_state["detection_result"] = result

        result = st.session_state.get("detection_result", {})
        if result.get("disease"):
            st.success(f"Disease: **{result['disease']}**")
        if result.get("confidence"):
            st.info(f"Confidence: {result['confidence']}%")
    else:
        st.warning("Please upload an image to begin detection.")

    st.markdown("</div>", unsafe_allow_html=True)

# 💡 TREATMENT ADVICE
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI-Generated Solutions")

    if "detection_result" in st.session_state:
        try:
            plant, disease = st.session_state["detection_result"]["disease"].split(" ", 1)
            key = f"{plant} {disease}"
        except ValueError:
            key = None

        if st.button("Show Treatment Advice") and key:
            catalog = load_disease_catalog()
            match = catalog[catalog["Disease Key"].str.lower() == key.lower()]

            if not match.empty:
                row = match.iloc[0]
                st.markdown("### Prevention")
                st.success(row["prevention"])
                st.markdown("### Treatment")
                st.success(row["treatment"])
            else:
                st.warning(f"No treatment record found for: {key}")
    else:
        st.warning("Please run disease detection first.")

    st.markdown("</div>", unsafe_allow_html=True)

# 📚 FULL DISEASE TREATMENT LIBRARY
with st.expander("Full Disease Treatment Library"):
    catalog = load_disease_catalog()
    query = st.text_input("Search for a disease (e.g., Cashew Gumosis)")
    results = catalog

    if query:
        results = catalog[catalog["Disease Key"].str.lower().str.contains(query.lower())]

    if results.empty:
        st.info("No matching diseases found.")
    else:
        for _, row in results.iterrows():
            st.subheader(row["Disease Key"])
            st.markdown(f"**Prevention:** {row['prevention']}")
            st.markdown(f"**Treatment:** {row['treatment']}")
            st.markdown("---")

# ❓ HELP & FAQ
with st.expander("Help & FAQ"):
    st.markdown("""
**Q1: What types of plants are supported?**  
✅ Currently supports tomato, maize, cassava, and cashew.

**Q2: How accurate is the detection?**  
📸 Accuracy depends on image clarity and model quality. Use high-resolution, close-up images for best results.

**Q3: Who can I contact for support?**  
📬 Email: [dtan3697@sdsu.edu](mailto:dtan3697@sdsu.edu)
""")

# 🔚 FOOTER
st.markdown("""
<hr style="margin-top: 3rem; margin-bottom: 1rem;">
<div style='text-align: center; color: gray; font-size: 0.85em'>
    © 2025 <strong>AI for Farmers</strong> — San Diego State University · 
    <a href="https://metabolismofcities-llab.org/" target="_blank" style="color: teal; text-decoration: none;">
        Visit MOCLL</a> · 
    <a href="mailto:dtan3697@sdsu.edu" style="color: teal; text-decoration: none;">Contact</a>
</div>
""", unsafe_allow_html=True)
