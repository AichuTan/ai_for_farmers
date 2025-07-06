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

    # Query LLM
    df.at[i, "prevention"] = index.query(f"How to prevent {plant} {disease} in 20 words?", llm=llm).strip()
    df.at[i, "treatment"]  = index.query(f"How to treat {plant} {disease} in 20 words?", llm=llm).strip()

    print(f"✔️ Processed {plant} {disease}")

# ─── Save Results ───────────────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"✅ Updated {CSV_PATH}")
