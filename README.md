## Innovate with AI Agents for Space Exploration

### Project Description
This project leverages AI agents to facilitate multilingual space exploration insights using NASA's API, speech-to-text processing, and advanced AI models. It enables users to ask space-related questions in various languages, with responses generated through intelligent AI agents trained on space data.

### Features
- **Multilingual AI Assistance**: Users can interact in different languages, and AI agents provide responses accordingly.
- **Speech-to-Text Processing**: Converts audio queries into text for further AI-driven analysis.
- **NASA API Integration**: Fetches real-time space-related data to enrich responses.
- **Vector Database (FAISS)**: Stores and retrieves relevant knowledge for contextual understanding.
- **Hugging Face Embeddings**: Utilizes pre-trained language models for efficient text embeddings.
- **Streamlit Interface**: Provides an interactive UI for seamless user experience.

### Technologies Used
- **Python**
- **Streamlit**
- **CrewAI**
- **Hugging Face Embeddings**
- **FAISS (Facebook AI Similarity Search)**
- **NASA API**
- **SpeechRecognition**
- **Pydub**
- **Requests**
- **LangChain Community Libraries**

### How It Works
1. Users select a language and input their space-related query (via text or audio file).
2. Audio inputs are transcribed using `SpeechRecognition`.
3. AI agents process the question using contextual knowledge from NASA API and FAISS.
4. The AI generates multilingual explanations based on predefined roles (Researcher & Educator agents).
5. Responses are displayed in the Streamlit interface.

### Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/AI-Space-Agents.git
   ```
2. Navigate to the project directory:
   ```sh
   cd AI-Space-Agents
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

### Future Improvements
- Enhance AI models for better language support.
- Implement real-time AI agent interactions.
- Expand data sources for richer space knowledge.



