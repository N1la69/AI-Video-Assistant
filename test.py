from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

from dotenv import load_dotenv
load_dotenv()

source = "https://www.youtube.com/watch?v=_Q-e_nczWqM"
language = "en" # Hindi language code = "hi" and English language code = "en"

chunks = process_input(source)


transcript = transcribe_all(chunks, language=language, translate=False)
print("\n" + "=" * 50)
print("TRANSCRIPT")
print("=" * 50)
print(transcript[:500] if len(transcript) > 500 else transcript)

title = generate_title(transcript)
summary = summarize(transcript)

print("\n" + "=" * 50)
print(f"MEETING TITLE: {title}")
print("=" * 50)
print("\nSUMMARY")
print("-" * 50)
print(summary)

action_items = extract_action_items(transcript)
key_decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 50)
print("ACTION ITEMS")
print("-" * 50)
print(action_items)

print("\n" + "=" * 50)
print("DECISIONS")
print("-" * 50)
print(key_decisions)

print("\n" + "=" * 50)
print("QUESTIONS")
print("-" * 50)
print(questions)