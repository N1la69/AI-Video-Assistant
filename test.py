from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from dotenv import load_dotenv

load_dotenv()

source = "https://www.youtube.com/watch?v=OfBBOgmxEeE"
language = "hinglish" # Change to "hinglish" to use Sarvam AI for Hinglish transcription

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)

print("\nFinal Transcript:\n")
print(transcript)