# app.py
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.llms import HuggingFaceHub
import requests
import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# Configuration - Using Streamlit Secrets
NASA_API_URL = f"https://api.nasa.gov/planetary/apod?api_key={st.secrets['NASA_API_KEY']}"
HF_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_REPO = "HuggingFaceH4/zephyr-7b-beta"

# Language Configuration
LANGUAGE_CODES = {
    'English': 'en-US',
    'Spanish': 'es-ES',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Chinese': 'zh-CN',
    'Arabic': 'ar-SA'
}

def speech_to_text(audio_file, language_code):
    """Convert uploaded audio file to text"""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name

        if not audio_path.endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path + ".wav"
            audio.export(wav_path, format="wav")
            audio_path = wav_path

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=language_code)
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return ""
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

def get_nasa_data():
    try:
        return requests.get(NASA_API_URL, timeout=10).json()
    except Exception as e:
        return {"error": f"NASA API Error: {str(e)}"}

def load_knowledge_base():
    try:
        loader = WebBaseLoader(["https://mars.nasa.gov/news/"])
        docs = loader.load()[:3]
        embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        return None

def setup_agents(language='en'):
    researcher = Agent(
        role="Multilingual Space Data Analyst",
        goal="Analyze and verify space-related data",
        backstory="Seasoned space scientist with expertise in multilingual data analysis",
        verbose=True,
        llm=HuggingFaceHub(
            repo_id=LLM_REPO,
            model_kwargs={"temperature": 0.3, "max_length": 512}
        ),
        memory=True
    )

    educator = Agent(
        role="Bilingual Science Communicator",
        goal=f"Explain space concepts in {language}",
        backstory=f"Fluent science communicator in {language}",
        verbose=True,
        llm=HuggingFaceHub(
            repo_id=LLM_REPO,
            model_kwargs={"temperature": 0.6, "max_length": 612}
        ),
        memory=True
    )
    return researcher, educator

def process_question(question, target_lang='en'):
    try:
        nasa_data = get_nasa_data()
        vector_store = load_knowledge_base()
        researcher, educator = setup_agents(target_lang)

        research_task = Task(
            description=f"Research: {question} | NASA Context: {nasa_data.get('explanation', '')}",
            agent=researcher,
            expected_output="3 verified technical points"
        )

        explain_task = Task(
            description=f"Explain in {target_lang} using simple terms",
            agent=educator,
            expected_output="2-paragraph answer",
            dependencies=[research_task]
        )

        crew = Crew(
            agents=[researcher, educator],
            tasks=[research_task, explain_task],
            verbose=True
        )
        return crew.kickoff()
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit Interface
st.title("🚀 Multilingual Space Agent")
st.markdown("### Ask space questions in any language!")

selected_lang = st.selectbox("Select Language", list(LANGUAGE_CODES.keys()))
lang_code = LANGUAGE_CODES[selected_lang].split('-')[0]

input_method = st.radio("Input Method", ["Text", "Audio File"])
question = ""

if input_method == "Text":
    question = st.text_input(f"Your space question in {selected_lang}:", "")
else:
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    if audio_file:
        with st.spinner("Processing audio..."):
            question = speech_to_text(audio_file, LANGUAGE_CODES[selected_lang])
            if question:
                st.text_area("Transcribed Text", value=question, height=100)

if question:
    with st.spinner("Analyzing with AI agents..."):
        answer = process_question(question, lang_code)
        st.markdown(f"### 🌍 Answer ({selected_lang}):")
        st.markdown(answer)

st.markdown("---")
st.markdown("*Powered by NASA API & Open Source AI*")
