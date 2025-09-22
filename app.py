# app.py
import io
import os
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

# Your inference utils (unchanged import path)
from utils.inference import detect_disease, get_all_disease_classes  # noqa: F401 (keep if you expose classes later)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI for Farmers", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS & CACHES
# ──────────────────────────────────────────────────────────────────────────────
def load_css(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

@st.cache_data
def load_disease_catalog():
    """Load CSV once; build normalized key + fast lookup map."""
    df = pd.read_csv("utils/data/disease_catalog.csv")
    # Expect first two cols: plant_type, disease
    df["plant_type"] = df.iloc[:, 0].astype(str).str.strip()
    df["disease"] = df.iloc[:, 1].astype(str).str.strip()
    df["Disease Key"] = df["plant_type"] + " " + df["disease"]
    df["key_norm"] = df["Disease Key"].str.lower()
    # Expect columns 'prevention' and 'treatment'
    lut = dict(zip(df["key_norm"], zip(df["prevention"], df["treatment"])))
    return df, lut

def init_state():
    for k, v in (
        ("input_image", None),
        ("input_video", None),
        ("detection_result", None),
        ("frame_second", 0),
        ("media_type", "Image"),
        ("advice_mode", "Both"),
    ):
        st.session_state.setdefault(k, v)

def extract_frame_from_video(uploaded_video, second: int = 0):
    """Grab a single frame from uploaded video at `second`."""
    try:
        import cv2
    except Exception as e:
        st.warning(f"Video processing not available ({e})")
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, int(second) * 1000)
        ok, frame = cap.read()
        cap.release()

        if ok and frame is not None:
            import numpy as np  # noqa: F401
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        else:
            st.warning("Couldn't read a frame at that time. Try a different second.")
    except Exception as e:
        st.warning(f"Error reading video: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return None



def pil_to_png_bytes(img) -> bytes:
    """Convert a PIL.Image or NumPy array to PNG bytes for download."""
    from PIL import Image
    import numpy as np
    buf = io.BytesIO()

    if isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, np.ndarray):
        # Ensure it's in uint8 format and convert to RGB if needed
        if img.dtype != "uint8":
            img = img.astype("uint8")
        if img.ndim == 2:  # grayscale
            pil_img = Image.fromarray(img, mode="L")
        else:
            pil_img = Image.fromarray(img)
    else:
        raise TypeError(f"Unsupported type for pil_to_png_bytes: {type(img)}")

    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def build_advice_markdown(key: str, prevention: str, treatment: str, mode: str) -> str:
    lines = [f"# {key}"]
    if mode in ("Prevention", "Both"):
        lines += ["", "## Prevention", "", prevention]
    if mode in ("Treatment", "Both"):
        lines += ["", "## Treatment", "", treatment]
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD CSS & INIT
# ──────────────────────────────────────────────────────────────────────────────
load_css("style.css")
init_state()

# ──────────────────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero' style='margin-bottom: 0.5rem'>
  <h1>AI for Farmers: Plant Disease Diagnosis</h1>
  <p>Detect plant diseases and receive AI-generated treatment recommendations</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# STATUS STRIP (Upload → Detect → Recommend)
# ──────────────────────────────────────────────────────────────────────────────
has_img = st.session_state.get("input_image") is not None
has_result = st.session_state.get("detection_result") is not None
st.markdown("""
<style>
.sticky-status { position: sticky; top: 0; z-index: 9; background: white; padding: .4rem 0 .2rem; }
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='sticky-status'>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"**① Upload** {'✅' if has_img else ''}")
    s2.markdown(f"**② Detect** {'✅' if has_result else ''}")
    s3.markdown(f"**③ Recommend** {'✅' if has_result else ''}")
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT: STICKY LEFT (Wizard rail) + RIGHT (Tabs)
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([0.32, 0.68], gap="large")

# =========================
# LEFT: Wizard (sticky)
# =========================
with left:
    st.markdown("<div class='sticky-left'>", unsafe_allow_html=True)


    with st.container(border=True):
        st.markdown("#### 🌿 Plant Disease Detector")

        # Guide line
        st.caption("1) Choose Image/Video  2) Upload/Sample frame  3) Run Detection")

        # Use a form to avoid re-runs while selecting
        with st.form("upload_form", clear_on_submit=False):
            media_type = st.radio(
                "Input type",
                ["Image", "Video"],
                horizontal=True,
                key="media_type",
            )

            if media_type == "Image":
                img_file = st.file_uploader(
                    "Browse image",
                    type=["jpg", "jpeg", "png"],
                    key="img_uploader",
                )
                if img_file:
                    # EXIF orientation fix + RGB
                    image = ImageOps.exif_transpose(Image.open(img_file).convert("RGB"))
                    st.session_state["input_image"] = image
                    st.session_state["input_video"] = None

            else:
                vid_file = st.file_uploader(
                    "Browse video",
                    type=["mp4", "mov", "avi", "mkv"],
                    key="vid_uploader",
                )
                if vid_file:
                    st.session_state["input_video"] = vid_file
                    st.session_state["input_image"] = None

                if st.session_state.get("input_video") is not None:
                    frame_second = st.slider(
                        "Pick a frame (seconds)",
                        min_value=0,
                        max_value=600,
                        value=int(st.session_state.get("frame_second", 0)),
                        step=1,
                        key="frame_second",
                    )
                    sample = st.form_submit_button("Sample frame")
                    if sample:
                        frame_img = extract_frame_from_video(
                            st.session_state["input_video"],
                            second=int(st.session_state["frame_second"]),
                        )
                        if frame_img:
                            st.session_state["input_image"] = ImageOps.exif_transpose(frame_img)

            # Detection button (only enabled when we have an image)
            run_detection = st.form_submit_button(
                "Run Detection",
                type="primary"
            )

        # Actually run detection only when pressed & image exists
        if run_detection and st.session_state.get("input_image") is not None:
            with st.spinner("Detecting disease…"):
                try:
                    result = detect_disease(st.session_state["input_image"])
                except Exception as e:
                    result = None
                    st.error(f"Detection failed: {e}")
                st.session_state["detection_result"] = result
            st.toast("Detection complete", icon="✅")

    # Helpful tips (collapsible)
    with st.expander("Tips for best results", expanded=False):
        st.markdown(
            "- Use a well-lit, in-focus close-up of the symptomatic area.\n"
            "- Avoid heavy motion blur (for video, sample a steady frame).\n"
            "- If the confidence is low, you can still browse advice in the Library tab."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================
# RIGHT: Tabs & Results
# =====================
with right:    
    with st.container(border=True):
    
        st.markdown("### 🔎 AI Based Treatment & Recommendations")

        # Quick chips (if available)
        result = st.session_state.get("detection_result") or {}
        chips = []
        if result.get("disease"):
            chips.append(f"<span class='chip'>Disease: {result['disease']}</span>")
        if result.get("confidence") is not None:
            conf = result["confidence"]
            # Keep the same % format as your original (already multiplied)
            chips.append(f"<span class='chip'>Confidence: {conf}%</span>")
        if chips:
            st.markdown(" ".join(chips), unsafe_allow_html=True)

        # Main tabs
        tab_preview, tab_detections, tab_advice, tab_library, tab_faq = st.tabs(
            ["🖼️ Preview", "📦 Detections", "💊 Advice", "📚 Library", "❓ FAQ"]
        )

        # ── Preview tab
        with tab_preview:
            preview = st.session_state.get("input_image")
            if preview is not None:
                st.image(preview, caption="Uploaded Image", width=400)
            elif st.session_state.get("input_video") is not None:
                st.video(st.session_state["input_video"])
                st.info("Sample a frame on the left, then run detection.")
            else:
                st.warning("Upload an image or video on the left to begin.")

        # ── Detections tab
        with tab_detections:
            annotated = result.get("annotated_image")
            if annotated is not None:
                st.image(annotated, caption="Detected Regions (YOLO)", width=400)
                # Optional: render a detections table if your result provides it
                dets = result.get("detections") or result.get("predictions")
                if dets:
                    # Expecting list of dicts or (label, score, bbox) tuples; render best-effort
                    rows = []
                    for d in dets:
                        if isinstance(d, dict):
                            rows.append({
                                "label": d.get("label") or d.get("class") or "",
                                "score": d.get("score") or d.get("confidence") or "",
                                "bbox": d.get("bbox") or "",
                            })
                        elif isinstance(d, (list, tuple)) and len(d) >= 2:
                            rows.append({"label": d[0], "score": d[1], "bbox": d[2] if len(d) > 2 else ""})
                    if rows:
                        st.markdown("**Detections**")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Annotated image will appear here after you run detection.")

        # ── Advice tab
        with tab_advice:
            catalog, lut = load_disease_catalog()

            # Determine the selected disease key
            selected_key = None
            if result and result.get("disease"):
                # Result like "Cashew gummosis"
                selected_key = result["disease"].strip()

            # If confidence is low or no match, allow manual override via selectbox
            if not selected_key or selected_key.lower() not in lut:
                st.info("No exact match from detection. You can still browse advice below.")
                all_keys = catalog["Disease Key"].tolist()
                selected_key = st.selectbox("Select a disease to view advice", options=all_keys)

            # Advice mode
            with st.form("advice_form"):
                advice_mode = st.radio(
                    "Show",
                    ["Prevention", "Treatment", "Both"],
                    horizontal=True,
                    key="advice_mode",
                )
                go_advice = st.form_submit_button("Generate Advice", type="primary")

            if go_advice and selected_key:
                key_norm = selected_key.lower()
                if key_norm in lut:
                    prevention, treatment = lut[key_norm]
                    # Expanders to reduce scrolling
                    if advice_mode in ("Prevention", "Both"):
                        with st.expander("Prevention", expanded=True):
                            st.success(prevention)
                    if advice_mode in ("Treatment", "Both"):
                        with st.expander("Treatment", expanded=True):
                            st.success(treatment)

                    # Downloads (annotated image + advice markdown)
                    cols = st.columns(2)
                    with cols[0]:
                        if result.get("annotated_image") is not None:
                            st.download_button(
                                "Download annotated image (PNG)",
                                data=pil_to_png_bytes(result["annotated_image"]),
                                file_name="annotated.png",
                                mime="image/png",
                                use_container_width=True,
                            )
                    with cols[1]:
                        md = build_advice_markdown(selected_key, prevention, treatment, advice_mode)
                        st.download_button(
                            "Save advice (Markdown)",
                            data=md,
                            file_name=f"{selected_key.replace(' ','_').lower()}_advice.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                else:
                    st.warning(f"No treatment record found for: {selected_key}")

        # ── Library tab
        with tab_library:
            catalog, _ = load_disease_catalog()
            query = st.text_input("Search (e.g., Cashew gummosis)")
            results = catalog if not query else catalog[catalog["Disease Key"].str.lower().str.contains(query.lower())]

            if results.empty:
                st.info("No matching diseases found.")
            else:
                for _, row in results.iterrows():
                    with st.expander(row["Disease Key"], expanded=False):
                        st.markdown(f"**Prevention**\n\n{row['prevention']}")
                        st.markdown("---")
                        st.markdown(f"**Treatment**\n\n{row['treatment']}")

        # ── FAQ tab
        with tab_faq:
            st.markdown("""
        **What plants are supported?** Tomato, Maize, Cassava, Cashew  
        **Accuracy tips:** Use high-res, well-lit, in-focus close-ups.  
        **Can I upload video?** Yes—sample a specific frame in the left panel.  
        **Support:** [thedebbietan@gmail.com](mailto:thedebbietan@gmail.com)
        """)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 1rem;">
<div style='text-align: center; color: gray; font-size: 0.85em; line-height: 1.6;'>
  © 2025 <strong>AI for Farmers</strong><br>
  Author: <a href="https://aichutan.github.io/" target="_blank" style="color: teal; text-decoration: none;">
  Aichu Tan</a><br>
  Supervisors: <strong>Dominico &amp; G. Fernandez</strong><br>
  Affiliation: 
  <a href="https://metabolismofcities-llab.org/metabolism-of-cities-living-lab-moc-llab/" target="_blank" style="color: teal; text-decoration: none;">
    Metabolism of Cities Living Lab
  </a>, 
  <a href="https://www.sdsu.edu/" target="_blank" style="color: teal; text-decoration: none;">
    San Diego State University
  </a><br>
  Presented at the <a href="https://onehealthconference.it/" target="_blank" style="color: teal; text-decoration: none;">
  4th International One Health Conference, Rome Italy 2025</a>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center; color: gray; font-size: 0.8em; margin-top: 1rem;'>
  Licensed under the <a href="https://opensource.org/licenses/MIT" target="_blank" style="color: teal; text-decoration: none;">MIT License</a>
</div>
""", unsafe_allow_html=True)


