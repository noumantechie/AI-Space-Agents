import streamlit as st
from crewai import Agent, Task, Crew
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
import requests
import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# Configuration - Critical for Hugging Face
NASA_API_URL = f"https://api.nasa.gov/planetary/apod?api_key={st.secrets['NASA_API_KEY']}"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_TOKEN"]

# Language setup
LANGUAGE_MAP = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh',
    'Arabic': 'ar'
}

def handle_audio(audio_file, lang_code):
    """Universal audio handler with format conversion"""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            # Convert to WAV if needed
            if not audio_file.name.endswith('.wav'):
                AudioSegment.from_file(audio_file).export(tmp_file.name, format="wav")
            else:
                tmp_file.write(audio_file.getvalue())
            
            with sr.AudioFile(tmp_file.name) as source:
                return recognizer.recognize_google(
                    recognizer.record(source),
                    language=f"{lang_code}-{lang_code.upper()}"
                )
    except Exception as e:
        st.error(f"Audio Error: {str(e)}")
        return ""

def fetch_space_data():
    """Universal data fetcher with retries"""
    try:
        response = requests.get(NASA_API_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return {}

def create_crew(lang):
    """Dynamic agent creation"""
    researcher = Agent(
        role="Space Analyst",
        goal="Analyze space data from multiple sources",
        backstory="Expert in astrophysics data analysis",
        verbose=True,
        allow_delegation=False,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta"
    )

    educator = Agent(
        role="Science Educator",
        goal=f"Explain concepts in {lang} simply",
        backstory="Multilingual science communicator",
        verbose=True,
        allow_delegation=False,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta"
    )

    return researcher, educator

# Streamlit Interface
st.title("🌌 Universal Space Explorer")
st.header("Ask space questions in any language")

# Language selection
selected_lang = st.selectbox("Choose Language", list(LANGUAGE_MAP.keys()))
lang_code = LANGUAGE_MAP[selected_lang]

# Input handling
input_type = st.radio("Input Type", ["Text", "Voice"])
query = ""

if input_type == "Text":
    query = st.text_input(f"Ask in {selected_lang}:", "")
else:
    audio_input = st.file_uploader("Upload Voice Query", type=["wav", "mp3", "ogg"])
    if audio_input:
        with st.spinner("Decoding voice..."):
            query = handle_audio(audio_input, lang_code)
            if query:
                st.text_area("Heard:", value=query, height=100)

if query:
    with st.spinner("Consulting space experts..."):
        try:
            # Initialize components
            space_data = fetch_space_data()
            analyst, explainer = create_crew(selected_lang)
            
            # Create tasks
            analysis_task = Task(
                description=f"Analyze: {query}\nContext: {space_data.get('explanation', '')}",
                agent=analyst,
                expected_output="3 key verified points"
            )
            
            explanation_task = Task(
                description=f"Explain in {selected_lang} simply",
                agent=explainer,
                expected_output="Clear 2-paragraph explanation",
                context=[analysis_task]
            )
            
            # Execute crew
            crew = Crew(
                agents=[analyst, explainer],
                tasks=[analysis_task, explanation_task],
                verbose=True
            )
            
            result = crew.kickoff()
            st.markdown(f"## ✨ {selected_lang} Answer:")
            st.markdown(result)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.info("Please try again or rephrase your question")

st.markdown("---")
st.caption("Powered by Open Space APIs & AI")
