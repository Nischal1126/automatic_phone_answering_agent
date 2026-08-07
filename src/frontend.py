"""
frontend.py

Streamlit UI for the RAG voice agent — single unified input row (text box
+ inline mic icon), ChatGPT-style. Talks to backend.py over HTTP.

Behavior:
    - Typing + submitting -> POST /chat/text  -> RAG only, text answer.
    - Clicking the mic     -> POST /chat/voice -> full STT -> RAG -> TTS
                               pipeline, spoken answer, autoplayed.

Only the mic path ever touches STT/TTS. Typing never does — the backend
enforces this split too (see backend.py); this UI just presents both
entry points in one row instead of separate sections.

Install:
    pip install streamlit audio-recorder-streamlit requests

Run (backend must already be running, e.g. `uvicorn backend:app --port 8000`):
    streamlit run frontend.py

Set BACKEND_URL as an environment variable if the backend isn't on
localhost:8000.
"""

import base64
import os
import tempfile

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Department Assistant", page_icon="🎓", layout="centered")

# Merge the text input and mic button into what reads as a single bar:
# zero gap between the two columns, mic vertically centered and nudged
# up against the input's right edge, background/border matched so there's
# no visible seam between them.
st.markdown(
    """
    <style>
    /* kill the default gap streamlit puts between columns in this row */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stChatInput"]) {
        gap: 0rem;
        align-items: center;
    }
    /* the mic column: no padding, vertically centered, pulled flush
       against the input box instead of floating with a gap */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stChatInput"])
        > div:last-child {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: -0.5rem;
        padding-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend client helpers
# ---------------------------------------------------------------------------

def ask_text(query: str) -> str:
    """RAG-only path. No STT, no TTS."""
    resp = requests.post(f"{BACKEND_URL}/chat/text", json={"query": query}, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"]


def ask_voice(audio_bytes: bytes) -> dict:
    """Full pipeline path: STT -> RAG -> TTS. Always returns spoken audio."""
    files = {"audio": ("clip.wav", audio_bytes, "audio/wav")}
    resp = requests.post(f"{BACKEND_URL}/chat/voice", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def save_base64_audio(b64_data: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(base64.b64decode(b64_data))
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎓 AI Department Assistant")
st.caption("Ask about programs, fees, or department info — by typing or by voice.")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Backend: {BACKEND_URL}")
    if st.button("Check backend health"):
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
            st.success(health)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backend unreachable: {exc}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "autoplay_index" not in st.session_state:
    st.session_state.autoplay_index = None

# Render chat history — only the most recent voice reply autoplays; older
# ones just show a normal player the user can click to replay.
autoplay_index = st.session_state.autoplay_index
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"], autoplay=(i == autoplay_index))
# Autoplay only fires once per new reply — clear it so a later rerun
# (e.g. from typing a new message) doesn't replay old audio.
st.session_state.autoplay_index = None

# --- Single unified input row: text box + inline mic ------------------
col_text, col_mic = st.columns([8, 1])
with col_text:
    text_query = st.chat_input("Type your question here...")
with col_mic:
    audio_bytes = audio_recorder(
        text="",
        icon_size="1.5x",
        recording_color="#e04141",
        neutral_color="#6c6c6c",
        key="mic_recorder",
    )

# ---------------------------------------------------------------------------
# Text path — RAG only
# ---------------------------------------------------------------------------

if text_query:
    st.session_state.messages.append({"role": "user", "content": text_query, "audio_path": None})
    with st.chat_message("user"):
        st.markdown(text_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = ask_text(text_query)
        except requests.RequestException as exc:
            response = f"⚠️ Couldn't reach the backend: {exc}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "audio_path": None})
    st.rerun()

# ---------------------------------------------------------------------------
# Voice path — full STT -> RAG -> TTS pipeline, only when audio was recorded
# ---------------------------------------------------------------------------

elif audio_bytes:
    # audio_recorder keeps returning the same bytes across reruns until a
    # new clip is recorded — guard on a hash so we don't reprocess the same
    # clip every time Streamlit reruns the script.
    audio_hash = hash(audio_bytes)
    if st.session_state.get("last_audio_hash") != audio_hash:
        st.session_state["last_audio_hash"] = audio_hash

        with st.spinner("Transcribing, thinking, and generating a spoken reply..."):
            try:
                result = ask_voice(audio_bytes)
            except requests.RequestException as exc:
                st.error(f"⚠️ Couldn't reach the backend: {exc}")
                result = None

        if result:
            st.session_state.messages.append(
                {"role": "user", "content": result["transcript"], "audio_path": None}
            )
            audio_path = None
            if result.get("audio_base64"):
                try:
                    audio_path = save_base64_audio(result["audio_base64"])
                except Exception:  # noqa: BLE001
                    audio_path = None
            else:
                st.warning("Got a text answer, but the spoken reply failed to generate.")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["response"], "audio_path": audio_path}
            )
            if audio_path:
                # index of the assistant message we just appended
                st.session_state.autoplay_index = len(st.session_state.messages) - 1
            st.rerun()