# AI for Farmers: Plant Disease Diagnosis & Advisory
A Streamlit web app that detects plant diseases from images (YOLO-based backend) and provides farmer-friendly prevention and treatment guidance from a curated literature catalog (RAG-style preprocessing, offline at runtime).
* Author: Aichu Tan, M.S. Big Data Analytics, San Diego State University
* Supervisors: Prof. Domenico Vito & Prof. Gabriel Fernandez
* Affiliation: Metabolism of Cities Living Lab, San Diego State University
* Presented at: 4th International One Health Conference, Rome, Italy 2025
* License: MIT

## **Features**
* Upload an image (or sample a frame from video) to detect plant diseases.
* View detections (labels, scores, optional annotated image).
* Browse prevention and treatment guidance from a local CSV catalog built from extension manuals and scientific literature.
* Search a library of diseases and download advice as Markdown.
* Works offline at runtime (advice served from CSV; no live LLM calls).

## **Demo**
Hugging Face (prototype): https://huggingface.co/spaces/aycu2004/ai_for_farmer
Note: Local setup is recommended for best performance and to use your own model weights.

## **How It Works**
1. Detection detect_disease(image) (in utils/inference.py) runs the detector (e.g., YOLOv8) and returns predicted disease, confidence, and optional annotated image/detections.
2. Advisory (Offline) The app loads utils/data/disease_catalog.csv once and maps disease keys to prevention and treatment text. Users can:
    * See auto-selected advice from the detection result, or
    * Manually choose a disease from the library.
    * Download annotated image and advice Markdown.
3. Video Frame Sampling (Optional) Upload a video, pick a timestamp, extract a frame, then run detection on that frame.

## **Data & Models**
* Default catalog supports Cashew, Cassava, Maize, Tomato with multiple diseases.
* Replace or extend disease_catalog.csv to add crops/diseases and their guidance.
* Train/retrain your model separately; point utils/inference.py to your weights.

## **Safety Notes (Advisory Output)**
* Advice prioritizes non-chemical measures first.
* If chemical options appear, they are generic (active ingredients) with reminders to:
    * Follow local regulations and label instructions.
    * Use appropriate PPE.
* Always validate guidance with local experts and extension services.

## **Citation**
If you use this repository, please cite:
Tan, A., Vito, D., & Fernandez, G. AI for Farmers: A YOLO and Language Model-Based Plant Disease Detection and Advisory System. Presented at the 4th International One Health Conference, Rome, Italy, 2025.

## **Acknowledgements**
* Metabolism of Cities Living Lab, Center for Human Dynamics in the Mobile Age, SDSU
* UENR (Ghana) for the CCMT dataset
* Ultralytics YOLO, LangChain, OpenAI, Streamlit, Roboflow, and Hugging Face teams

## **License**
This project is licensed under the MIT License. See LICENSE for details.

