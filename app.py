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

# Configuration - Critical fix for Hugging Face
NASA_API_URL = f"https://api.nasa.gov/planetary/apod?api_key={st.secrets['NASA_API_KEY']}"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACE_API_TOKEN"]

LANGUAGE_CODES = {
    'English': 'en-US',
    'Spanish': 'es-ES',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Chinese': 'zh-CN',
    'Arabic': 'ar-SA'
}

def speech_to_text(audio_file, language_code):
    """Convert audio to text with improved error handling"""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Handle non-WAV files
            if not audio_file.name.endswith('.wav'):
                audio = AudioSegment.from_file(audio_file)
                audio.export(tmp_file.name, format="wav")
            else:
                tmp_file.write(audio_file.getvalue())
            
            # Process audio
            with sr.AudioFile(tmp_file.name) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data, language=language_code)
                
    except sr.UnknownValueError:
        st.error("Could not understand audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech recognition error: {e}")
        return ""
    finally:
        if 'tmp_file' in locals() and os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

def get_nasa_data():
    """Fetch NASA data with timeout handling"""
    try:
        response = requests.get(NASA_API_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"NASA API Error: {str(e)}")
        return {}

def load_knowledge_base():
    """Load space knowledge base with error recovery"""
    try:
        loader = WebBaseLoader(["https://mars.nasa.gov/news/"])
        docs = loader.load()[:3]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Knowledge base error: {str(e)}")
        return None

def setup_agents(language='en'):
    """Configure AI agents with simplified setup"""
    researcher = Agent(
        role="Space Research Analyst",
        goal="Analyze and verify space-related data from NASA sources",
        backstory="Expert in astrophysics data analysis with multilingual capabilities",
        verbose=True,
        allow_delegation=False,
        memory=True,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta"
    )

    educator = Agent(
        role="Science Educator",
        goal="Explain complex space concepts in simple terms",
        backstory="Multilingual science communicator with space science background",
        verbose=True,
        allow_delegation=False,
        memory=True,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta"
    )

    return researcher, educator

def process_question(question, lang_code='en'):
    """Process questions through AI crew with enhanced error handling"""
    try:
        nasa_data = get_nasa_data()
        researcher, educator = setup_agents(lang_code)

        research_task = Task(
            description=f"Research: {question}\nNASA Context: {nasa_data.get('explanation', '')}",
            agent=researcher,
            expected_output="3 verified technical points with sources",
            async_execution=True
        )

        explain_task = Task(
            description=f"Explain in {lang_code} using simple analogies and examples",
            agent=educator,
            expected_output="2-paragraph explanation suitable for non-experts",
            context=[research_task]
        )

        crew = Crew(
            agents=[researcher, educator],
            tasks=[research_task, explain_task],
            verbose=True
        )

        return crew.kickoff()
    
    except Exception as e:
        st.error(f"AI Processing Error: {str(e)}")
        return "Sorry, we encountered an error processing your question. Please try again."

# Streamlit Interface
st.title("🚀 Multilingual Space Agent")
st.markdown("### Ask space questions in any language!")

# Language Selection
selected_lang = st.selectbox("Select Language", list(LANGUAGE_CODES.keys()))
lang_code = LANGUAGE_CODES[selected_lang].split('-')[0]

# Input Handling
input_method = st.radio("Input Method", ["Text", "Audio"])
question = ""

if input_method == "Text":
    question = st.text_input(f"Ask your space question in {selected_lang}:", "")
else:
    audio_file = st.file_uploader("Upload audio question", type=["wav", "mp3", "ogg"])
    if audio_file:
        with st.spinner("Processing audio..."):
            question = speech_to_text(audio_file, LANGUAGE_CODES[selected_lang])
            if question:
                st.text_area("Transcribed Question", value=question, height=100)

# Processing and Output
if question:
    with st.spinner("Analyzing with space experts..."):
        try:
            answer = process_question(question, lang_code)
            st.markdown(f"### 🌍 Answer ({selected_lang}):")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.info("Please check your internet connection and try again.")

st.markdown("---")
st.markdown("*Powered by NASA API & Open Source AI*")
