from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_questions

load_dotenv()


def run_pipeline(source: str, language: str = "en") -> dict:
    print("Starting AI Video Assistant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language=language)
    print(f"Raw Transcription (first 300 characters): {transcript[:300]}")

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


if __name__ == "__main__":
    source = input("Enter YouTube URL or Local File Path: ").strip()
    language = input("Language (en/hi): ").strip() or "en"
    result = run_pipeline(source, language)

    print("\n" + "=" * 50)
    print(f"Title: {result['title']}")
    print(f"\nSummary: {result['summary']}")
    print(f"\nAction items: {result['action_items']}")
    print(f"\nKey Decisions: {result['key_decisions']}")
    print(f"\nOpen Questions: {result['open_questions']}")
    print("=" * 50)

    # Chat with Meeting AI via RAG
    print("\nChat with your meeting (type 'exit' to quit)\n")
    rag_chain = result['rag_chain']
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_questions(rag_chain, question)
        print(f"\nAssistant: {answer}\n")