import yt_dlp
import os
import subprocess

DOWNLOAD_DIR = "downloads"
FFMPEG_PATH = os.path.join("ffmpeg", "bin")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        "ffmpeg_location": FFMPEG_PATH,

        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],

        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(url, download=True)

        filename = (
            os.path.splitext(
                ydl.prepare_filename(info)
            )[0] + ".wav"
        )

    return filename


def convert_to_wav(input_path: str) -> str:

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    ffmpeg_exe = os.path.join(
        FFMPEG_PATH,
        "ffmpeg.exe"
    )

    command = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ]

    subprocess.run(command, check=True)

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    chunk_duration = chunk_minutes * 60

    output_dir = os.path.splitext(wav_path)[0] + "_chunks"

    os.makedirs(output_dir, exist_ok=True)

    ffmpeg_exe = os.path.join(
        FFMPEG_PATH,
        "ffmpeg.exe"
    )

    output_pattern = os.path.join(
        output_dir,
        "chunk_%03d.wav"
    )

    command = [
        ffmpeg_exe,
        "-i", wav_path,

        "-f", "segment",
        "-segment_time", str(chunk_duration),

        "-c", "copy",

        output_pattern
    ]

    subprocess.run(command, check=True)

    chunks = sorted([
        os.path.join(output_dir, file)
        for file in os.listdir(output_dir)
        if file.endswith(".wav")
    ])

    return chunks


def process_input(source:str) -> list:
    if source.startswith("http"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    print(f"Processing complete. Generated {len(chunks)} chunks.")
    return chunks

