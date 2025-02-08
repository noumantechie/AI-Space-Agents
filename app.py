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

# Configuration
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
    """Convert audio to text using Google's speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            if audio_file.name.endswith('.wav'):
                tmp_file.write(audio_file.getvalue())
            else:
                audio = AudioSegment.from_file(audio_file)
                audio.export(tmp_file.name, format="wav")
            
            with sr.AudioFile(tmp_file.name) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data, language=language_code)
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return ""
    finally:
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

def get_nasa_data():
    """Fetch space data from NASA API"""
    try:
        return requests.get(NASA_API_URL, timeout=10).json()
    except Exception as e:
        return {"error": str(e)}

def load_knowledge_base():
    """Load space knowledge base"""
    try:
        loader = WebBaseLoader(["https://mars.nasa.gov/news/"])
        docs = loader.load()[:3]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Knowledge base error: {str(e)}")
        return None

def setup_agents(language='en'):
    """Configure AI agents"""
    researcher = Agent(
        role="Space Research Analyst",
        goal="Analyze and verify space-related data",
        backstory="Expert in space science data analysis",
        verbose=True,
        allow_delegation=False,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta",
        llm_kwargs={"temperature": 0.3, "max_length": 512}
    )

    educator = Agent(
        role="Science Educator",
        goal="Explain complex concepts simply",
        backstory="Multilingual science communicator",
        verbose=True,
        allow_delegation=False,
        llm="huggingface/HuggingFaceH4/zephyr-7b-beta",
        llm_kwargs={"temperature": 0.6, "max_length": 612}
    )

    return researcher, educator

def process_question(question, lang_code='en'):
    """Process user question through AI crew"""
    try:
        nasa_data = get_nasa_data()
        researcher, educator = setup_agents(lang_code)

        research_task = Task(
            description=f"Research: {question}\nNASA Data: {nasa_data.get('explanation', '')}",
            agent=researcher,
            expected_output="3 verified technical points"
        )

        explain_task = Task(
            description=f"Explain in {lang_code} using simple terms",
            agent=educator,
            expected_output="2-paragraph explanation",
            context=[research_task]
        )

        crew = Crew(
            agents=[researcher, educator],
            tasks=[research_task, explain_task],
            verbose=True
        )

        return crew.kickoff()
    except Exception as e:
        return f"Processing error: {str(e)}"

# Streamlit Interface
st.title("🚀 Multilingual Space Agent")
st.markdown("### Ask space questions in any language!")

selected_lang = st.selectbox("Select Language", list(LANGUAGE_CODES.keys()))
lang_code = LANGUAGE_CODES[selected_lang].split('-')[0]

input_method = st.radio("Input Method", ["Text", "Audio"])
question = ""

if input_method == "Text":
    question = st.text_input(f"Ask in {selected_lang}:", "")
else:
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg"])
    if audio_file:
        with st.spinner("Processing audio..."):
            question = speech_to_text(audio_file, LANGUAGE_CODES[selected_lang])
            if question:
                st.text_area("Transcribed Text", value=question, height=100)

if question:
    with st.spinner("Analyzing with AI crew..."):
        answer = process_question(question, lang_code)
        st.markdown(f"### 🌍 Answer ({selected_lang}):")
        st.markdown(answer)

st.markdown("---")
st.markdown("*Powered by NASA API & Open Source AI*")
