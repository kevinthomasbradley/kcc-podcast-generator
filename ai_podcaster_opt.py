"""
ai_podcaster_v2.py

Author: Kevin Bradley
Version: 1.0.0
Date: 2025-06-26
Description: 
    Streamlit app for generating podcast scripts using LLMs and synthesizing audio with Kokoro TTS.
"""

import re
from pathlib import Path
import numpy as np
import soundfile as sf
import streamlit as st
from kokoro import KPipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# --- Constants and Templates ---

TURN_RE = re.compile(r"^\*\*(.+?):\*\*\s*$", re.MULTILINE)
VOICE_MAP = {
    "Lev": "am_michael",
    "Mila": "af_bella",
}

SUMMARY_TEMPLATE = """
Summarize the following text by highlighting the key points from the text.
Text: {text}
"""

JUDGE_TEMPLATE = """
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

SCRIPT_TEMPLATE = """
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

# --- LLM Utilities ---

def get_llm(model_name="deepseek-r1:8b"):
    """Return an LLM model instance."""
    return ChatOllama(model=model_name)

def generate_topic_script(topic: str, model, name="Kevins Coffee & Code", speakerA="Lev", speakerB="Mila") -> str:
    """Generate a podcast script for the given topic using an LLM."""
    prompt = ChatPromptTemplate.from_template(SCRIPT_TEMPLATE)
    chain = prompt | model
    script = chain.invoke({"name": name, "topic": topic, "speakerA": speakerA, "speakerB": speakerB})
    return clean_reasoning_from_text(script.content)

def judge_script(script: str, model) -> str:
    """Use an LLM to judge the quality and formatting of the podcast script."""
    prompt = ChatPromptTemplate.from_template(JUDGE_TEMPLATE)
    chain = prompt | model
    result = chain.invoke({"script": script})
    return result.content.strip()

def clean_reasoning_from_text(text: str) -> str:
    """Remove <think>...</think> tags and extra whitespace from the script."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def parse_script_into_turns(script: str) -> list:
    """Parse the markdown script into a list of conversation turns."""
    turns_list = []
    lines = script.splitlines()
    i = 0
    while i < len(lines):
        title_match = TURN_RE.match(lines[i])
        if title_match:
            speaker = title_match.group(1).strip()
            dialogue = []
            i += 1
            while i < len(lines) and not TURN_RE.match(lines[i]):
                if lines[i].strip():
                    dialogue.append(lines[i].strip())
                i += 1
            full_text = " ".join(dialogue)
            turns_list.append({"speaker": speaker, "text": full_text})
        else:
            i += 1
    return turns_list

# --- Audio Utilities ---

def clean_directory(directory: Path):
    """Remove all files in the given directory."""
    for f in directory.glob("*"):
        if f.is_file():
            f.unlink()

def generate_audio_parts(pipeline, turns: list, out_dir: Path) -> None:
    """Generate audio files for each conversation turn using Kokoro TTS."""
    for idx, turn in enumerate(turns, start=1):
        speaker = turn["speaker"].split()[0]
        voice = VOICE_MAP.get(speaker, "af_heart")
        gen = pipeline(turn["text"], voice=voice, speed=1.3, split_pattern=None)
        for j, (_, _, audio) in enumerate(gen):
            fname = out_dir / f"{idx:02d}_{speaker}_{j}.wav"
            sf.write(fname, audio, 24_000)
            print("saved", fname)

def combine_audio_parts(out_dir: Path) -> Path:
    """Combine all generated .wav files in the output directory into a single audio file."""
    wav_files = sorted(out_dir.glob("*.wav"))
    if wav_files:
        data, samplerate = sf.read(wav_files[0])
        combined_audio = data.copy()
        for wav_file in wav_files[1:]:
            data, _ = sf.read(wav_file)
            combined_audio = np.concatenate((combined_audio, data))
        output_file = out_dir / "combined_audio.wav"
        sf.write(output_file, combined_audio, samplerate)
        print(f"Combined audio saved to {output_file}")
        return output_file
    else:
        print("No .wav files found in the output directory.")

# --- Streamlit UI ---

def main():
    # Initialize session state
    if "script_text" not in st.session_state:
        st.session_state["script_text"] = ""
    if "turns" not in st.session_state:
        st.session_state["turns"] = []
    if "judge_result" not in st.session_state:
        st.session_state["judge_result"] = ""

    st.title("Kevins Coffee & Code")
    topic_text = st.text_area("Enter the topic of the podcast:", max_chars=100)
    script_button = st.button("Generate Script")

    llm = get_llm()

    if script_button and topic_text:
        generated_script = generate_topic_script(topic_text, llm)
        turns = parse_script_into_turns(generated_script)
        turns_text = "\n\n".join([f"**{t['speaker']}:**  \n{t['text']}" for t in turns])
        st.session_state["script_text"] = turns_text
        st.session_state["turns"] = turns
        # Optionally judge the script
        st.session_state["judge_result"] = judge_script(generated_script, llm)

    if st.session_state["judge_result"]:
        st.subheader("AI Judge Verdict")
        st.info(st.session_state["judge_result"])

    script_text = st.text_area(
        "The generated script for the topic:",
        value=st.session_state["script_text"],
        height=400
    )

    generate_podcast_button = st.button(
        "Generate Podcast",
        disabled=not bool(st.session_state["script_text"].strip()) or st.session_state["judge_result"].startswith("FAIL")
    )

    if generate_podcast_button:
        turns = st.session_state["turns"]
        st.write("Generating the audio for the podcast")
        pipeline = KPipeline(lang_code="a")
        out_dir = Path("audios")
        out_dir.mkdir(exist_ok=True)
        clean_directory(out_dir)
        generate_audio_parts(pipeline, turns, out_dir)
        output_file = combine_audio_parts(out_dir)
        st.write("Audio file generated successfully")
        if output_file:
            with open(output_file, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")

if __name__ == "__main__":
    main()