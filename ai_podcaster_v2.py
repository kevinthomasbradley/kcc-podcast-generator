"""
ai_podcaster_v2.py

Author: Kevin Bradley
Version: 1.0.0
Date: 2025-06-26
Description: 
    Streamlit app for generating podcast scripts using LLMs and synthesizing audio with Kokoro TTS.
    Features:
    - LLM-based script generation for two speakers
    - Markdown parsing and cleaning
    - Audio synthesis and concatenation
    - Interactive Streamlit UI for topic input and audio playback
"""
import os  # For file and directory operations
import re  # For regular expressions (used in text cleaning and parsing)
from textwrap import dedent
from xml.sax.saxutils import escape
from pathlib import Path

import numpy as np  # For numerical operations, especially array concatenation
import soundfile as sf  # For reading and writing audio files
import streamlit as st  # For building the web UI
from kokoro import KPipeline  # For text-to-speech synthesis
from langchain_core.prompts import ChatPromptTemplate  # For prompt templating
from langchain_ollama import ChatOllama  # For LLM chat interface

# --- Prompt Templates ---

# Template for summarizing text using the LLM
summary_template = """
Summarize the following text by highlighting the key points from the text.
Text: {text}
"""

# Template for judging the script quality and formatting
judge_template = """
You are an expert podcast script reviewer. Review the following script for:
- Conversational flow and engagement
- Proper markdown formatting (each turn starts with **Speaker:**)
- No extraneous effects or tags
- No more than 20 turns
- Both speakers participate equally

Provide a verdict: "PASS" if the script is optimal and formatted correctly, or "FAIL" with a brief explanation of issues.

Script:
{script}
"""

# Template for generating a podcast script with two speakers
script_template = """
Acting as a podcast scriptwriter, generate a conversational script for a podcast episode based on the given topic. Format the output based on two people, each taking a turn, the first is called {speakerA} and the second is called {speakerB}. 
They are both tech enthusiasts and will discuss the topic in a friendly, engaging manner. Use markdown for formatting, with each speaker's name contained within asterik blocks followed by their dialogue on a new line. Do not add any effects for instance **(EFFECT)**, just keep the script as simple text. In total there should be no more than 20 turns in the conversation.
The podcast name is {name}.
Topic: {topic}
Example Output:
**{speakerA}:**  
Hey everyone, welcome back to {name}! I’m {speakerA}, and today we’re diving into something that’s been buzzing all over tech circles lately—Large Language Models.  

**{speakerB}:**  
Hi folks, I’m {speakerB}! Yeah, {speakerA}, it’s wild how quickly these models went from research labs to everyday tools. I mean, who would’ve thought a few years ago we’d be using AI to write emails, generate code, even brainstorm marketing campaigns?

**{speakerA}:**  
Exactly! And for anyone new to this—Large Language Models, or LLMs, are basically advanced AI systems trained on massive amounts of text data. Think of them as extremely overachieving autocomplete engines with a flair for sounding human.

**{speakerB}:**  
That’s a great way to put it. But they’re not just parroting back what they’ve read. They actually predict what comes next in a sentence based on patterns they’ve seen across billions of words. It’s like predictive text on steroids.

**{speakerA}:**  
Right—and one of the most famous examples is ChatGPT, which is built on OpenAI’s GPT models. But there’s a whole ecosystem now: Claude, Gemini, LLaMA, Mistral… Everyone’s got one.

**{speakerB}:**  
And while the capabilities are amazing, there’s also some serious stuff to consider—bias, misinformation, ethical concerns. These models are only as good—or as flawed—as the data they’re trained on.

**{speakerA}:**  
Totally. You’ve got to use them responsibly. They’re great assistants, but not always reliable truth-tellers. You still need human judgment in the loop.

**{speakerB}:**  
But if used thoughtfully, they can save time, amplify creativity, and help level the playing field for folks without access to big teams or resources.

**{speakerA}:**  
Well said. We’ll definitely be unpacking more on that in future episodes. But for today, that’s our quick take on Large Language Models.  

**{speakerB}:**  
Thanks for listening, everyone! Stay curious—and caffeinated.

**{speakerA}:**  
Catch you next time on Suzannes Frequencies.
"""

# --- Regular Expressions and Voice Mapping ---

# Regex to match lines that start with "**" followed by a name and ":**"
TURN_RE = re.compile(r"^\*\*(.+?):\*\*\s*$", re.MULTILINE)

# Mapping of speaker names to Kokoro TTS voice IDs
VOICE_MAP = {
    "Lev": "am_michael",   # American‑male, solid quality
    "Mila" : "af_bella",   # American‑female, high quality
}

# --- LLM Model Initialization ---

# Initialize the LLM model for script generation
model = ChatOllama(model="deepseek-r1:8b")

# Initialize the LLM model for analyzing and judging the script
#judge_model = ChatOllama(model="deepseek-r1:8b")

# --- AI as a Judge ---

#def judge_script(script: str) -> str:
#    """
#    Use an LLM to judge the quality and formatting of the podcast script.
#
#    Args:
#        script (str): The script to review.
#
#    Returns:
#        str: The LLM's verdict and feedback.
#    """
#    prompt = ChatPromptTemplate.from_template(judge_template)
#    chain = prompt | judge_model
#    result = chain.invoke({"script": script})
#    return result.content.strip()

# --- Podcast Script Generation ---

def generate_topic_script(topic: str) -> str:
    """
    Generate a podcast script for the given topic using an LLM.

    Args:
        topic (str): The topic for the podcast episode.

    Returns:
        str: The cleaned, generated script as a string.
    """
    name = "Kevins Coffee & Code"  # Podcast name
    speakerA = "Lev"  # First speaker
    speakerB = "Mila"   # Second speaker

    prompt = ChatPromptTemplate.from_template(script_template)
    chain = prompt | model

    # Generate the script using the LLM
    script = chain.invoke({"name": name, "topic": topic, "speakerA": speakerA, "speakerB": speakerB})
    
    # Remove any <think> tags and clean up the script text
    cleaned_script = clean_reasoning_from_text(script.content)
    
    # Print this cleaned script to the console for debugging
    print(f"Generated script for topic '{topic}':\n{cleaned_script}")
    
    return cleaned_script

def clean_reasoning_from_text(text: str) -> str:
    """
    Remove <think>...</think> tags and extra whitespace from the script.

    Args:
        text (str): The script text to clean.

    Returns:
        str: The cleaned script text.
    """
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def parse_script_into_turns(script: str) -> list:
    """
    Parse the markdown script into a list of conversation turns.

    Args:
        script (str): The script in markdown format.

    Returns:
        list: A list of dictionaries, each with 'speaker' and 'text' keys.
    """
    turns_list = []
    lines = script.splitlines()
    i = 0
    while i < len(lines):
        title_match = TURN_RE.match(lines[i])
        if title_match:
            speaker = title_match.group(1).strip()
            # Collect subsequent non-blank lines as that speaker’s text
            dialogue = []
            i += 1
            while i < len(lines) and not TURN_RE.match(lines[i]):
                if lines[i].strip():  # skip true blank lines
                    dialogue.append(lines[i].strip())
                i += 1
            full_text = " ".join(dialogue)
            turns_list.append({"speaker": speaker, "text": full_text})
        else:
            i += 1

    for t in turns_list:
        print(f"{t['speaker']}: {t['text']}")

    return turns_list

# --- Audio Generation ---

def generate_audio_parts(pipeline, turns: list, out_dir: Path) -> None:
    """
    Generate audio files for each conversation turn using Kokoro TTS.

    Args:
        pipeline: The Kokoro TTS pipeline object.
        turns (list): List of conversation turns (dicts with 'speaker' and 'text').
        out_dir (Path): Directory to save the generated audio files.
    """
    for idx, turn in enumerate(turns, start=1):
        speaker = turn["speaker"].split()[0]         # "David", "Maya"
        voice   = VOICE_MAP.get(speaker, "af_heart") # fallback voice
        
        # Generate one clip (Kokoro may chunk internally – we loop anyway)
        gen = pipeline(turn["text"], voice=voice, speed=1.3, split_pattern=None)
        
        for j, (_, _, audio) in enumerate(gen):
            fname = out_dir / f"{idx:02d}_{speaker}_{j}.wav"
            sf.write(fname, audio, 24_000)            # 24 kHz native rate
            print("saved", fname)

def combine_audio_parts(out_dir: Path) -> Path:
    """
    Combine all generated .wav files in the output directory into a single audio file.

    Args:
        out_dir (Path): Directory containing the .wav files.

    Returns:
        Path: Path to the combined audio file, or None if no files found.
    """
    wav_files = sorted(out_dir.glob("*.wav"))

    # Read the first file to get the sample rate
    if wav_files:
        data, samplerate = sf.read(wav_files[0])
        combined_audio = data.copy()

        # Read and concatenate the rest of the files
        for wav_file in wav_files[1:]:
            data, _ = sf.read(wav_file)
            combined_audio = np.concatenate((combined_audio, data))

        # Write the combined audio to a new file
        output_file = out_dir / "combined_audio.wav"
        sf.write(output_file, combined_audio, samplerate)
        print(f"Combined audio saved to {output_file}")
        return output_file
    else:
        print("No .wav files found in the output directory.")

# --- Streamlit UI Section ---

# Initialize session state for script_text and turns if not already set
if "script_text" not in st.session_state:
    st.session_state["script_text"] = ""
if "turns" not in st.session_state:
    st.session_state["turns"] = []

# Set the title of the web app
st.title("Kevins Coffee & Code")

# Text area for user to input the topic of the podcast
topic_text = st.text_area("Enter the topic of the podcast:", max_chars=100)

# Button to trigger script generation
script_button = st.button("Generate Script")

# Main logic: when the button is pressed and text is provided
if script_button and topic_text:
    # Generate the script and parse into turns
    generated_script = generate_topic_script(topic_text)
    turns = parse_script_into_turns(generated_script)

    # Format turns as markdown for display
    turns_text = "\n\n".join([f"**{t['speaker']}:**  \n{t['text']}" for t in turns])

    # Update session state with the script and turns
    st.session_state["script_text"] = turns_text
    st.session_state["turns"] = turns

    # Judge the script and store the result
    # judge_result = judge_script(generated_script)
    # st.session_state["judge_result"] = judge_result

# Display the judge's verdict
# if "judge_result" in st.session_state:
#    st.subheader("AI Judge Verdict")
#    st.info(st.session_state["judge_result"])

# Display the generated script in a text area
script_text = st.text_area(
    "The generated script for the topic:",
    value=st.session_state["script_text"],
    height=400
)

# Add a button that is only enabled when script_text is not empty
generate_podcast_button = st.button(
    "Generate Podcast",
    disabled=not bool(st.session_state["script_text"].strip())
)

#generate_podcast_button = st.button(
#    "Generate Podcast",
#    disabled=(
#        not bool(st.session_state["script_text"].strip()) or
#        ("judge_result" in st.session_state and st.session_state["judge_result"].startswith("FAIL"))
#    )
#)

# Generate and display the podcast audio when the button is pressed
if generate_podcast_button:
    # Retrieve turns from session state
    turns = st.session_state["turns"]  
    st.write("Generating the audio for the podcast")

    # Initialize Kokoro TTS pipeline
    pipeline = KPipeline(lang_code="a")   
    out_dir = Path("audios")
    out_dir.mkdir(exist_ok=True)

    # Generate audio for each turn and combine
    generate_audio_parts(pipeline, turns, out_dir)
    output_file = combine_audio_parts(out_dir)

    st.write("Audio file generated successfully")

    if output_file:
        # Display and play the .wav file in Streamlit
        with open(output_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/wav")