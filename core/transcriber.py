import whisper
import os

# Add local FFmpeg to PATH
FFMPEG_PATH = os.path.abspath(
    os.path.join("ffmpeg", "bin")
)

os.environ["PATH"] += os.pathsep + FFMPEG_PATH

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)

_model = None


def load_model():

    global _model

    if _model is None:

        print(
            f"Loading Whisper model: "
            f"{WHISPER_MODEL}"
        )

        _model = whisper.load_model(
            WHISPER_MODEL
        )

        print("Model loaded successfully.")

    return _model


def transcribe_chunk(
    chunk_path: str,
    language: str = None,
    translate: bool = False
) -> str:

    model = load_model()

    task = (
        "translate"
        if translate
        else "transcribe"
    )

    result = model.transcribe(
        chunk_path,

        language=language,

        task=task,

        fp16=False
    )

    return result["text"]


def transcribe_all(
    chunks: list,

    language: str = None,

    translate: bool = False
) -> str:

    full_transcript = ""

    print(
        f"Using Whisper "
        f"({WHISPER_MODEL})"
    )

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i+1}/{len(chunks)}"
        )

        text = transcribe_chunk(
            chunk_path=chunk,

            language=language,

            translate=translate
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()