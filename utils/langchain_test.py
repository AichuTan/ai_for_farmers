import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in yoloenv/.env")
print("🔑 Loaded OPENAI_API_KEY: ✅ FOUND")

# ─── Imports ────────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# ─── Config ─────────────────────────────────────────────────────────────────
repo_root     = Path(__file__).resolve().parent  # or define explicitly
CSV_PATH      = repo_root / "data" / "disease_catalog.csv"
PDF_DIR       = repo_root / "data" / "disease_library"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ─── Initialize LLM and Embeddings ──────────────────────────────────────────
llm = ChatOpenAI(temperature=0)
embedder = OpenAIEmbeddings(openai_api_key=API_KEY)

# ─── Load Disease Catalog ───────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["prevention"] = ""
df["treatment"]  = ""

# ─── Process Each Disease ───────────────────────────────────────────────────
for i, row in df.iterrows():
    plant = row["plant_type"]
    disease = row["disease"]

    file_name = f"{plant.lower()}_{disease.lower().replace(' ', '_')}.pdf"
    pdf_fp = PDF_DIR / file_name

    if not pdf_fp.exists():
        print(f"⚠️  Skipping {plant} {disease}: PDF not found ({file_name})")
        continue

    # Load and split PDF
    raw_docs = PyPDFLoader(str(pdf_fp)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(raw_docs)

    # Build vector index
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embedder
    ).from_documents(docs)


    # ✅ Query LLM with clear, farmer-focused prompts (INSIDE the loop)
    df.at[i, "prevention"] = index.query(
        f"""Provide farmer-friendly PREVENTION advice for {plant} {disease}.
        Rules:
        - Write only bullet points.
        - Maximum 6 bullets, maximum 120 words total.
        - Start each bullet with a clear action verb (e.g., Remove, Prune, Plant).
        - Focus only on practical, non-chemical actions (sanitation, pruning, resistant varieties, spacing, rotation).
        - Avoid vague terms like 'regularly monitor' — make actions specific and measurable (e.g., "Inspect leaves weekly").
        - Do NOT use brand names or trade names.
        - Avoid jargon and technical terms. Use simple farmer language.""",
        llm=llm
    ).strip()

    df.at[i, "treatment"] = index.query(
        f"""Provide farmer-friendly TREATMENT advice for {plant} {disease}.

    Output EXACTLY in this structure (headings + bullets only, no extra text):
    Non-chemical:
    - ...
    - ...

    Chemical (last option):
    - <class or active> — when: <clear trigger>; interval: <X–Y days>; note: Follow label instructions and wear PPE.

    Seek help when:
    - ...

    Rules:
    - Use only bullet points under each heading (max TOTAL 6 bullets across all sections).
    - Give AT LEAST 2 non-chemical bullets before any chemical item.
    - Non-chemical examples: remove infected material, pruning, spacing/airflow, rotation, sanitation, beneficial insects, biocontrols.
    - Chemical items MUST use active ingredient or chemical class ONLY (e.g., copper fungicide, sulfur, azadirachtin, Trichoderma); NEVER brand or trade names.
    - EXCLUDE banned/restricted actives: carbendazim, benomyl, paraquat, monocrotophos, aldrin, dieldrin, endosulfan.
    - For each chemical bullet include: a clear WHEN trigger and an interval (e.g., every 10–14 days), and the note 'Follow label instructions and wear PPE.'
    - Make actions specific and measurable (e.g., 'Prune infected shoots within 24–48 hours', 'Inspect leaves weekly').
    - Avoid jargon; use simple farmer language.""",
        llm=llm
    ).strip()


    print(f"✔️ Processed {plant} {disease}")




# ─── Save Results ───────────────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"✅ Updated {CSV_PATH}")
