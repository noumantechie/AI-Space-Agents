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
import litellm

# Configuration
NASA_API_URL = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"
HF_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "huggingface/HuggingFaceH4/zephyr-7b-beta"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Set Hugging Face API token in environment
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN 

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
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name

        # Convert to WAV if necessary
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
    prompts = {
        'en': "Explain space concepts clearly in English",
        'es': "Explica conceptos espaciales en español",
        'fr': "Expliquez les concepts spatiaux en français",
        'de': "Erklären Sie Raumfahrtkonzepte auf Deutsch",
        'zh': "用中文清楚解释空间概念",
        'ar': "اشرح مفاهيم الفضاء باللغة العربية"
    }

    researcher = Agent(
        role="Multilingual Space Analyst",
        goal="Analyze and validate space information",
        backstory="Expert in multilingual space data analysis with NASA mission experience.",
        verbose=True,
        llm=LLM_MODEL,
        memory=True
    )

    educator = Agent(
        role="Bilingual Science Educator",
        goal=f"Explain complex concepts in {language} using simple terms",
        backstory=f"Multilingual science communicator specializing in {language} explanations.",
        verbose=True,
        llm=LLM_MODEL,
        memory=True
    )

    return researcher, educator

def process_question(question, target_lang='en'):
    try:
        nasa_data = get_nasa_data()
        vector_store = load_knowledge_base()
        researcher, educator = setup_agents(target_lang)

        research_task = Task(
            description=f"""Research: {question}
            NASA Context: {nasa_data.get('explanation', '')}
            Language: {target_lang}""",
            agent=researcher,
            expected_output="3 verified technical points",
            output_file="research.md"
        )

        explain_task = Task(
            description=f"Explain in {target_lang} using simple terms and analogies",
            agent=educator,
            expected_output="2-paragraph answer in requested language",
            context=[research_task]
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
st.title("🚀 COSMOLAB (Multilingual Space Agent)")
st.markdown("### Ask space questions in any language!")

# Single language selection for both input and output
selected_lang = st.selectbox("Select Language", list(LANGUAGE_CODES.keys()))
lang_code = LANGUAGE_CODES[selected_lang].split('-')[0]  # Extract base language code

# Input Method
input_method = st.radio("Input Method", ["Text", "Audio File"])

question = ""
if input_method == "Text":
    question = st.text_input(f"Your space question in {selected_lang}:", "")
else:
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])

    if audio_file is not None:
        with st.spinner("Processing audio..."):
            question = speech_to_text(audio_file, LANGUAGE_CODES[selected_lang])
            if question:
                st.text_area("Transcribed Text", value=question, height=100)

if question:
    with st.spinner("Analyzing with AI agents..."):
        try:
            # Use litellm to get AI response
            response = litellm.completion(
                model=LLM_MODEL,  # Correct model format
                api_key=HUGGINGFACE_API_TOKEN,  # Ensure API Key is passed
                messages=[{"role": "user", "content": question}]
            )
            answer = response['choices'][0]['message']['content']

            st.markdown(f"### 🌍 Answer ({selected_lang}):")
            st.markdown(answer)

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("Powered by NASA API & Open Source AI")
