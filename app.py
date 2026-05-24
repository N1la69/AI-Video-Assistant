import os
import tempfile
from dotenv import load_dotenv

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import (
    build_rag_chain,
    ask_questions
)

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

# ---------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: 600;
}

.block-container {
    padding-top: 2rem;
}

.chat-box {
    background-color: #1e1e1e;
    padding: 1rem;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# Pipeline Function
# ---------------------------------------------------

def run_pipeline(source: str, language: str = "en") -> dict:

    chunks = process_input(source)

    transcript = transcribe_all(
        chunks,
        language=language
    )

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain
    }

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("🎙️ AI Meeting Assistant")

st.markdown("""
Analyze meetings from:
- 📺 YouTube URLs
- 📁 Uploaded audio/video files

Features:
- Whisper transcription
- AI summaries
- Action item extraction
- Key decision extraction
- Meeting Q&A with RAG
""")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    language = st.selectbox(
        "Select Language",
        options=[
            ("English", "en"),
            ("Hindi", "hi"),
            ("Bengali", "bn")
        ],
        format_func=lambda x: x[0]
    )[1]

    st.markdown("---")

    st.info(
        "Supports local Whisper transcription "
        "and multilingual meeting analysis."
    )

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

tab1, tab2 = st.tabs([
    "📺 YouTube URL",
    "📁 Upload File"
])

source = None

with tab1:

    youtube_url = st.text_input(
        "Enter YouTube URL"
    )

    if youtube_url:
        source = youtube_url

with tab2:

    uploaded_file = st.file_uploader(
        "Upload audio/video file",
        type=[
            "mp3",
            "wav",
            "mp4",
            "m4a",
            "webm"
        ]
    )

    if uploaded_file is not None:

        temp_dir = tempfile.mkdtemp()

        temp_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        source = temp_path

# ---------------------------------------------------
# Process Button
# ---------------------------------------------------

if st.button("🚀 Process Meeting"):

    if not source:

        st.warning(
            "Please provide a YouTube URL "
            "or upload a file."
        )

    else:

        with st.spinner(
            "Processing meeting... "
            "This may take several minutes."
        ):

            try:

                result = run_pipeline(
                    source,
                    language
                )

                st.session_state.result = result

                st.success(
                    "Meeting processed successfully!"
                )

            except Exception as e:

                st.error(f"Error: {str(e)}")

# ---------------------------------------------------
# Results Section
# ---------------------------------------------------

if st.session_state.result:

    result = st.session_state.result

    st.markdown("---")

    st.header(f"📝 {result['title']}")

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------

    st.subheader("📄 Meeting Summary")

    st.write(result["summary"])

    # -----------------------------------------------
    # Insights
    # -----------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("✅ Action Items")

        st.write(result["action_items"])

    with col2:

        st.subheader("📌 Key Decisions")

        st.write(result["key_decisions"])

    st.subheader("❓ Open Questions")

    st.write(result["open_questions"])

    # -----------------------------------------------
    # Transcript
    # -----------------------------------------------

    with st.expander("📜 Full Transcript"):

        st.write(result["transcript"])

    # -----------------------------------------------
    # Export Section
    # -----------------------------------------------

    st.subheader("📥 Export")

    export_text = f"""
TITLE:
{result['title']}

SUMMARY:
{result['summary']}

ACTION ITEMS:
{result['action_items']}

KEY DECISIONS:
{result['key_decisions']}

OPEN QUESTIONS:
{result['open_questions']}

TRANSCRIPT:
{result['transcript']}
"""

    st.download_button(
        label="⬇️ Download TXT",
        data=export_text,
        file_name="meeting_report.txt",
        mime="text/plain"
    )

    # -----------------------------------------------
    # Chat with Meeting
    # -----------------------------------------------

    st.markdown("---")

    st.header("💬 Chat with Your Meeting")

    user_question = st.text_input(
        "Ask a question about the meeting"
    )

    if st.button("Ask"):

        if user_question.strip():

            with st.spinner("Thinking..."):

                answer = ask_questions(
                    result["rag_chain"],
                    user_question
                )

                st.session_state.chat_history.append(
                    {
                        "question": user_question,
                        "answer": answer
                    }
                )

    # -----------------------------------------------
    # Chat History
    # -----------------------------------------------

    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"""
            <div class="chat-box">
            <b>🧑 You:</b><br>
            {chat['question']}
            <br><br>
            <b>🤖 Assistant:</b><br>
            {chat['answer']}
            </div>
            <br>
            """,
            unsafe_allow_html=True
        )