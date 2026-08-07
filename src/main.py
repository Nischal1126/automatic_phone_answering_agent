import os
import io
import asyncio
import tempfile

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.retrievers.self_query.chroma import ChromaTranslator
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
import speech_recognition as sr
from faster_whisper import WhisperModel
import edge_tts
import pygame

CHROMA_DB_DIR = r"C:\Users\nisch\OneDrive\Desktop\automated voice agent\chroma_db"
COLLECTION_NAME = "doai_site"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 7

WHISPER_MODEL_SIZE = "base.en"          # tiny.en / base.en / small.en
EDGE_TTS_VOICE = "en-US-AriaNeural"     # try en-US-GuyNeural, en-GB-SoniaNeural, etc.
SILENCE_SECONDS_TO_END_TURN = 1.0       # how long you can pause before it decides you're done talking
EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "stop"}


def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )


def load_llm():
    return ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key = "",
)


metadata_field_info = [
    AttributeInfo(
        name="degree_level",
        description="The academic degree level the content applies to. One of ['Undergraduate', 'Graduate']",
        type="string",
    ),
    AttributeInfo(
        name="category",
        description="The type of content. One of ['Fee structure', 'academic_program']",
        type="string",
    ),
    AttributeInfo(
        name="department",
        description="The department the content belongs to, e.g. 'Artificial Intelligence'",
        type="string",
    ),
    AttributeInfo(
        name="source_file",
        description="Original source markdown filename this chunk came from",
        type="string",
    ),
]

document_content_description = (
    "Information about academic programs, fee structures, and department "
    "details for the Department of Artificial Intelligence at Kathmandu University"
)


def build_self_query_retriever(llm, vectordb):
    return SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        search_kwargs={"k": TOP_K},
        verbose=True,
    )


PROMPT_TEMPLATE = """You are a helpful phone assistant for the AI Department at the college.
Answer the caller's question using ONLY the context below. Keep the answer short,
natural, and conversational — 1 to 3 sentences — since it will be read aloud over the phone.
If the answer isn't in the context, say you don't have that information and offer to
connect the caller to the department office. Do not make anything up.

Context:
{context}

Caller's question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def safe_retrieve(query: str, retriever, vectordb):
    try:
        seed_docs = retriever.invoke(query)
        if not seed_docs:
            print("  [self-query returned 0 results - falling back to unfiltered search]")
            seed_docs = vectordb.similarity_search(query, k=TOP_K)
    except Exception as e:
        print(f"  [self-query failed ({type(e).__name__}: {e}) - falling back to unfiltered search]")
        seed_docs = vectordb.similarity_search(query, k=TOP_K)

    return seed_docs


MAX_SOURCE_DOCS = 2


def expand_to_full_documents(seed_docs, vectordb, max_source_docs=MAX_SOURCE_DOCS):
    source_files = []
    for d in seed_docs:
        sf = d.metadata.get("source_file")
        if sf and sf not in source_files:
            source_files.append(sf)
        if len(source_files) >= max_source_docs:
            break

    if not source_files:
        return seed_docs

    expanded = []
    for sf in source_files:
        result = vectordb.get(where={"source_file": sf})
        pairs = list(zip(result["documents"], result["metadatas"]))
        pairs.sort(key=lambda p: p[1].get("chunk_index", 0))
        expanded.extend(Document(page_content=text, metadata=meta) for text, meta in pairs)

    return expanded


def build_rag_chain():
    embeddings = load_embeddings()
    vectordb = load_vectorstore(embeddings)
    llm = load_llm()
    retriever = build_self_query_retriever(llm, vectordb)

    rag_chain = (
        {
            "context": RunnableLambda(
                lambda q: format_docs(
                    expand_to_full_documents(safe_retrieve(q, retriever, vectordb), vectordb)
                )
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def answer_query(rag_chain, query: str) -> str:
    return rag_chain.invoke(query)


# ---------------------------------------------------------------------------
# Speech-to-text: faster-whisper (offline)
# ---------------------------------------------------------------------------

def load_stt_model():
    # compute_type="int8" keeps CPU inference fast with a small accuracy cost
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe(stt_model, audio: sr.AudioData) -> str:
    wav_bytes = audio.get_wav_data()
    segments, _ = stt_model.transcribe(io.BytesIO(wav_bytes), language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def listen_for_utterance(recognizer, mic, phrase_time_limit=20):
    """
    Blocks until the caller starts speaking, then continues recording
    until they pause for SILENCE_SECONDS_TO_END_TURN — no keypress
    involved. Returns None if nothing usable came through.
    """
    with mic as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            return None
    return audio


# ---------------------------------------------------------------------------
# Text-to-speech: edge-tts (natural neural voice) + pygame playback
# ---------------------------------------------------------------------------

async def _synthesize(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    await communicate.save(out_path)


def speak(text: str):
    """
    Synthesizes and plays speech, blocking until playback finishes. This
    is deliberately synchronous/blocking so the mic isn't listening while
    the agent's own voice is playing (which would otherwise get picked
    up as the next 'utterance').
    """
    print(f"Response (spoken): {text}\n")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        asyncio.run(_synthesize(text, tmp_path))
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.unload()
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    print("Loading RAG pipeline...")
    chain, retriever = build_rag_chain()

    print(f"Loading speech-to-text model (faster-whisper: {WHISPER_MODEL_SIZE})...")
    stt_model = load_stt_model()

    pygame.mixer.init()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = SILENCE_SECONDS_TO_END_TURN
    mic = sr.Microphone()

    with mic as source:
        print("Calibrating for ambient noise, stay quiet for a second...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("Ready. Just start talking — say 'exit' or 'goodbye' to quit (Ctrl+C also works).\n")

    while True:
        audio = listen_for_utterance(recognizer, mic)
        if audio is None:
            continue

        query = transcribe(stt_model, audio)
        if not query:
            print("  [couldn't make out any speech, try again]")
            continue

        print(f"Caller said: {query}")

        if query.strip().lower().rstrip(".!?") in EXIT_PHRASES:
            speak("Goodbye.")
            break

        response = answer_query(chain, query)
        speak(response)