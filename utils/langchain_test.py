#!/usr/bin/env python3
# utils/langchain_test.py

from pathlib import Path
from dotenv import load_dotenv
import os

repo_root = Path(__file__).parent.parent
dotenv_path = repo_root / "yoloenv" / ".env"

print(f"📄 Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 Loaded OPENAI_API_KEY: {'✅ FOUND' if api_key else '❌ MISSING'}")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in yoloenv/.env")

import pandas as pd

# 2) Updated/community imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI       # reads OPENAI_API_KEY automatically

# ─── CONFIG ────────────────────────────────────────────────────────────────
API_KEY       = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in yoloenv/.env")

CSV_PATH      = repo_root / "data" / "disease_catalog.csv"
PDF_DIR       = repo_root / "data" / "disease_library"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
# ────────────────────────────────────────────────────────────────────────────

# 3) Initialize LLM + Embeddings
llm      = ChatOpenAI(temperature=0)
embedder = OpenAIEmbeddings(openai_api_key=API_KEY)

# 4) Read your catalog and add empty cols
df = pd.read_csv(CSV_PATH)
df["prevention"] = ""
df["treatment"]  = ""

# 5) Loop through each disease
for i, row in df.iterrows():
    plant = row["plant_type"]
    disease = row["disease"]
    
    file_name = f"{plant.lower()}_{disease.lower().replace(' ', '_')}.pdf"
    pdf_fp = PDF_DIR / file_name

    if not pdf_fp.exists():
        print(f"⚠️  Skipping {plant} {disease}: PDF not found ({file_name})")
        continue

    # 5a) Load + chunk
    raw_docs = PyPDFLoader(str(pdf_fp)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(raw_docs)

    # 5b) Build vector index
    index = VectorstoreIndexCreator(
        vectorstore_cls=DocArrayInMemorySearch,
        embedding=embedder
    ).from_documents(docs)

    # 5c) Run queries
    prev_q = f"How to prevent {plant} {disease} in 20 words?"
    treat_q = f"How to treat {plant} {disease} in 20 words?"

    df.at[i, "prevention"] = index.query(prev_q, llm=llm).strip()
    df.at[i, "treatment"]  = index.query(treat_q, llm=llm).strip()

    print(f"✔️ Processed {plant} {disease}")


# 6) Save back to CSV
df.to_csv(CSV_PATH, index=False)
print(f"✅ Updated {CSV_PATH}")
