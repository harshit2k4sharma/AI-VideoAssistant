# AI Video Assistant 🎬

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LLM Orchestration](https://img.shields.io/badge/Orchestrator-LangChain%20LCEL-121212.svg)](https://python.langchain.com/)
[![LLM Provider](https://img.shields.io/badge/LLM-Mistral%20AI-FD5722.svg)](https://mistral.ai/)
[![Speech Model](https://img.shields.io/badge/Speech-Sarvam%20%7C%20Whisper-7c3aed.svg)](https://www.sarvam.ai/)

**AI Video Assistant** is an advanced meeting intelligence and video analysis platform that transcribes, summarizes, and extracts key insights from local audio/video files or YouTube links. 

Featuring a bilingual engine (supporting **English** and **Hinglish**), it transcribes content, processes long transcripts via a **Map-Reduce summarizer**, extracts high-value structured entities (Action Items, Key Decisions, Open Questions), and establishes an interactive **Conversational RAG (Retrieval-Augmented Generation)** interface to let you "chat" with your video recordings.

---

## ✨ Features

*   **Bilingual Speech-to-Text Pipeline:**
    *   **English:** Uses OpenAI's **Whisper** model locally for cost-effective transcription.
    *   **Hinglish (Hindi-English mix):** Leverages **Sarvam AI's speech-to-text-translate API** (`saaras:v2.5` model) to transcribe and translate Hinglish speech directly into structured English text.
    *   *Custom Splicing Engine:* Seamlessly bypasses Sarvam's strict 30-second audio constraint by automatically slicing longer files into 25-second pieces with overlapping buffers, transcribing them, and stitching the results back together.
*   **Map-Reduce Summarizer:** Uses LangChain LCEL and Mistral AI (`mistral-small-latest`) to digest and summarize multi-hour meetings without hitting context window bounds or losing meeting context.
*   **Actionable Intelligence Extraction:** Automatically extracts:
    *   **Action Items:** Consisting of Task descriptions, assigned Owners, and Deadlines.
    *   **Key Decisions:** Documenting all pivot points and collective team agreements.
    *   **Open Questions:** Highlighting unresolved issues or topics requiring follow-up.
*   **Conversational RAG (Retrieval-Augmented Generation):**
    *   Chunks transcripts and creates dense vector indexes locally using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
    *   Persists and queries vector data using a local `Chroma` database.
    *   Enables fact-checked, context-constrained Q&A directly related to your meeting.
*   **Premium Front-End Workspace Dashboard:** A custom-styled Streamlit application using sleek dark-mode aesthetics, custom glassmorphism panels, glowing cards, and live pipeline progress indicators.

---

## 🏗️ Architecture Flow

```
[YouTube Link / Local Video] 
         │
         ▼ (yt-dlp / pydub)
  [16kHz Mono WAV Chunks] 
         │
         ├─────────────────────── Language Route ────────────────────────┐
         ▼ (English)                                                     ▼ (Hinglish)
  [Local Whisper small]                                            [Sarvam AI API]
         │                                                       (25s pieces auto-spliced)
         │                                                               │
         └──────────────────────► Stitched Transcript ◄──────────────────┘
                                         │
                                         ▼ (LangChain LCEL Chains)
                ┌────────────────────────┼────────────────────────┐
                ▼ (Map-Reduce)           ▼ (Extraction)           ▼ (Vector DB RAG)
         [Meeting Summary]     [Action Items / Decisions]   [Chroma DB + HF Embed]
                                                                  │
                                                                  ▼
                                                          [RAG Chat Interface]
```

---

## 🛠️ Prerequisites & Installation

### 1. Install FFmpeg
This application relies on `pydub` and `yt-dlp` for audio processing and conversion, which require **FFmpeg** to be installed and added to your system's PATH.

*   **Windows (Using Chocolatey):**
    ```bash
    choco install ffmpeg
    ```
    *Or download directly from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin/` directory to your system environment variables.*
*   **macOS (Using Homebrew):**
    ```bash
    brew install ffmpeg
    ```
*   **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```

### 2. Set Up the Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harshit2k4sharma/AI-VideoAssistant.git
    cd AI-VideoAssistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r Requirements.txt
    ```

---

## ⚙️ Configuration

1.  **Create your environment file:**
    Duplicate the `.env.example` file in the root folder and rename it to `.env`:
    ```bash
    cp .env.example .env
    ```

2.  **Add your API Keys:**
    Open the newly created `.env` file and insert your API keys:
    ```ini
    MISTRAL_API_KEY="your_mistral_api_key"
    SARVAM_API_KEY="your_sarvam_api_key"
    ```
    *(Note: `.env` is automatically ignored by git and will never be committed to protect your keys.)*

---

## 🚀 Running the Application

You can interact with the AI Video Assistant through either the rich Streamlit dashboard UI or a simple CLI terminal client.

### Option A: Launch the Streamlit Dashboard (Recommended)
Run the Streamlit app to start the premium workspace interface:
```bash
streamlit run app.py
```
*Your browser will automatically open to `http://localhost:8501` showing the dashboard.*

### Option B: CLI Terminal Interface
For running pipelines directly from your command line:
```bash
python main.py
```
*Follow the interactive prompts to enter a YouTube link or local file path, select your transcription language, view generated summaries, and chat with your transcript in the terminal.*

---

## 💡 Assessment Verification Notes
The primary assessment deliverables are included in the repository:
*   **Full Technical Write-up:** See [PROJECT_SUBMISSION.md](PROJECT_SUBMISSION.md) in the root directory. It contains explicit answers for:
    *   The core problem solved by the AI Video Assistant.
    *   Detailed architectural breakdowns with system interactions.
    *   The deep engineering challenge of handling Sarvam's 30-second API limitation and the custom chunk splicing engine built to solve it.
