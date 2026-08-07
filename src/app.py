"""
app.py

Streamlit chat UI (text + voice) for the RAG voice agent, ChatGPT-style.

- Text input: standard chat box.
- Voice input: click the mic to record, click again to stop; the clip is
  transcribed locally with faster-whisper (loaded from main.py).
- Every answer is shown as text AND spoken back with edge-tts (autoplay),
  with a sidebar toggle to turn voice replies off if you just want text.

This is push-to-talk, not always-on background listening — browsers don't
allow silent background mic capture, so click-to-record is the standard
pattern for a web chat UI (this is also how ChatGPT's own web voice mode
works under the hood).

Assumes main.py (in the same folder) exposes:
    build_rag_chain() -> (chain, retriever)
    answer_query(chain, query: str) -> str
    load_stt_model() -> faster_whisper.WhisperModel

If your file/functions are named differently, just adjust the import
below.

Install requirements (on top of what main.py already needs):
    pip install streamlit audio-recorder-streamlit edge-tts

Run with:
    streamlit run app.py
"""

import asyncio
import io
import os
import tempfile

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import edge_tts

from main import build_rag_chain, answer_query, load_stt_model

EDGE_TTS_VOICE = "en-US-AriaNeural"

st.set_page_config(page_title="AI Department Assistant", page_icon="🎓", layout="centered")


# ---------------------------------------------------------------------------
# Cached resources — loaded once per server process, not per interaction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_rag_chain():
    return build_rag_chain()


@st.cache_resource(show_spinner="Loading speech-to-text model...")
def get_stt_model():
    return load_stt_model()


chain, retriever = get_rag_chain()
stt_model = get_stt_model()


# ---------------------------------------------------------------------------
# STT / TTS helpers
# ---------------------------------------------------------------------------

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    segments, _ = stt_model.transcribe(io.BytesIO(audio_bytes), language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


async def _synthesize(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(out_path)


def synthesize_speech(text: str) -> str:
    """Returns the path to a temp mp3 file containing the spoken response."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    asyncio.run(_synthesize(text, tmp.name))
    return tmp.name


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎓 AI Department Assistant")
st.caption("Ask about programs, fees, or department info — by typing or by voice.")

with st.sidebar:
    st.header("Settings")
    voice_replies = st.toggle("Speak answers out loud", value=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"])

# Input row: text box + mic button side by side
col_text, col_mic = st.columns([6, 1])
with col_mic:
    audio_bytes = audio_recorder(
        text="",
        icon_size="2x",
        recording_color="#e04141",
        neutral_color="#6c6c6c",
        key="mic_recorder",
    )
with col_text:
    text_query = st.chat_input("Type your question here...")

# Figure out what the user actually asked, from whichever input fired
query = None
if text_query:
    query = text_query
elif audio_bytes:
    with st.spinner("Transcribing..."):
        query = transcribe_audio_bytes(audio_bytes)
    if not query:
        st.warning("Couldn't make out any speech in that recording — try again.")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_query(chain, query)
        st.markdown(response)

        audio_path = None
        if voice_replies:
            with st.spinner("Generating voice..."):
                audio_path = synthesize_speech(response)
            st.audio(audio_path, autoplay=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "audio_path": audio_path}
    )