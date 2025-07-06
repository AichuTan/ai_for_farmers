# utils/langchain_agent.py

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load and normalize CSV
df = pd.read_csv("data/plant_disease_prevention_treatment_FAKE.csv")
df["Plant"] = df["Plant"].str.strip().str.lower()
df["Disease"] = df["Disease"].str.strip().str.lower()

def get_llm(agent_choice):
    if agent_choice == "GPT-3.5 (OpenAI)":
        return ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
    elif agent_choice == "Mistral 7B (HF)":
        return HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"temperature": 0.4, "max_new_tokens": 240}
        )
    elif agent_choice == "Zephyr 7B (HF)":
        return HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"temperature": 0.4, "max_new_tokens": 240}
        )
    else:
        raise ValueError("Unknown LLM agent selected.")


def generate_treatment_summary(plant, disease, agent_choice):
    try:
        row = df[
            (df["Plant"] == plant.strip().lower()) &
            (df["Disease"] == disease.strip().lower())
        ].iloc[0]
        treatment = row["Treatment"]
        prevention = row["Prevention"]
    except IndexError:
        return "❌ No treatment information found for this plant and disease."

    llm = get_llm(agent_choice)

    base_prompt = (
        f"Plant: {plant}\nDisease: {disease}\n"
        f"Treatment: {treatment}\nPrevention: {prevention}"
    )

    # For HuggingFace chat models, use Human/Assistant style
    if "HF" in agent_choice:
        prompt_text = (
            "Human: Based on the following info, write a 20-word description of the treatment and prevention:\n\n"
            f"{base_prompt}\nAssistant:"
        )
    else:
        # for OpenAI
        prompt_text = (
            f"Based on the following info, write a 20-word description of the treatment and prevention:\n\n"
            f"{base_prompt}"
        )

    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else response


def translate_summary(summary, language, agent_choice):
    if language.lower() == "english":
        return summary

    llm = get_llm(agent_choice)

    prompt = ChatPromptTemplate.from_template(
        'Translate the following English sentence into {language}:\n\n"{summary}"'
    )
    response = llm.invoke(prompt.format(language=language, summary=summary))
    return response.content if hasattr(response, "content") else response
