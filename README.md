# Gemini Live Voice Agent

A real-time, full-duplex conversational **Voice Agent** powered by **Google Gemini Live API** (`google-genai`) and **LangChain** (`langchain-google-genai`).

Features native speech-to-speech audio streaming over WebSockets, Voice Activity Detection (VAD) with instant user interruption, tool calling (weather, math, notes, time), a terminal CLI mode, and an interactive Web UI with animated audio waveforms.

---

## Architecture Overview

```
                                      ┌─────────────────────────────────┐
                                      │   User (Microphone / Audio In)  │
                                      └────────────────┬────────────────┘
                                                       │ 16kHz PCM
                                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          Voice Agent Client                                            │
│                                                                                                        │
│   ┌────────────────────────────────────────┐            ┌──────────────────────────────────────────┐   │
│   │ CLI Interface (apps/cli_agent.py)      │    OR      │ Web Interface (apps/web_agent.py)        │   │
│   │ • sounddevice RawInputStream           │            │ • FastAPI WebSocket (/ws/voice)          │   │
│   │ • sounddevice RawOutputStream          │            │ • Web Audio API + HTML5 Visualizer       │   │
│   └───────────────────┬────────────────────┘            └────────────────────┬─────────────────────┘   │
└───────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────┘
                        │                                                      │
                        └──────────────────────┬───────────────────────────────┘
                                               │ Full-Duplex WebSocket
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │     GeminiLiveSession (Services Layer)       │
                        │     • VAD & Interruption Handling            │
                        │     • Real-time Transcription Stream         │
                        │     • Extensible Function Calling Tools      │
                        └──────────────────────┬───────────────────────┘
                                               │ Bidirectional Audio Stream
                                               ▼
                        ┌──────────────────────────────────────────────┐
                        │           Google Gemini Live API             │
                        │     Model: gemini-3.1-flash-live-preview     │
                        │     Voice: Puck / Charon / Aoede             │
                        └──────────────────────────────────────────────┘
```

---

## Directory Structure

```
Voice Agent/
├── apps/                         # Executable Entrypoints
│   ├── cli_agent.py              # Terminal Voice Assistant (Mic & Speaker)
│   └── web_agent.py              # FastAPI Web & WebSocket Voice Server
├── src/                          # Core Domain & Engine Layers
│   ├── core/                     # Configuration, Audio Specs (16kHz / 24kHz)
│   │   ├── config.py
│   │   └── __init__.py
│   ├── audio/                    # Hardware Audio I/O Drivers
│   │   ├── player.py             # 24kHz PCM speaker playback & instant interruption
│   │   ├── recorder.py           # 16kHz PCM microphone stream
│   │   └── __init__.py
│   ├── tools/                    # Function Calling Tools
│   │   ├── system_tools.py       # Time, Weather, Calculator, Notepad
│   │   └── __init__.py
│   └── services/                 # AI Engine
│       ├── gemini_live_session.py# Native Gemini Live WebSocket Speech-to-Speech
│       ├── langchain_agent.py    # LangChain ChatGoogleGenerativeAI Agent
│       └── __init__.py
├── web/                          # High-Fidelity Web UI
│   └── index.html                # Dark-mode dashboard with waveform visualizer
├── tests/                        # Automated Test Suites
│   └── test_agent.py             # Core unit tests
├── .env.example                  # Environment template
├── requirements.txt              # Pinned deterministic dependencies
└── README.md
```

---

## Quickstart

### 1. Set your Gemini API Key
Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey):

```bash
# In your terminal:
export GEMINI_API_KEY="AIzaSyYourGeneratedKeyHere"

# OR create a .env file:
cp .env.example .env
# Edit .env and paste your key
```

### 2. Activate the Virtual Environment
```bash
source .venv/bin/activate
```

---

## Running the Agent

### Mode A: Terminal Voice Assistant (CLI)
Speak directly into your computer's microphone and listen through your speakers:

```bash
python apps/cli_agent.py
```
- Real-time live transcriptions printed directly in your terminal.
- Instant interruption: Simply start speaking while Gemini is talking, and the playback will cut off immediately.

### Mode B: Browser Web UI with Waveform Visualizer
Launch the local web server:

```bash
python apps/web_agent.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser:
- Click **"Start Voice Chat"** and allow microphone access.
- Watch the live audio waveform and conversation transcript.

---

## Running Tests

```bash
python tests/test_agent.py
```
