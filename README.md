# Kevins Coffee & Code Podcast Generator

A small project that utilizes an LLM (deepseek-r1:8b) via Langchain to generate a podcast script on a particular topic, this script is then converted into audio using text to speech library Kokoro, and exposes this functionality via Streamlit.

# Prerequisites

```bash
ollama pull deepseek-r1:8b
```

```bash
# Use 3.11 as some dependency issues with later versions
python3.11 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

# Run
```bash
streamlit run ai_podcaster_v2.py
```