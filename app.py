import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import pipeline
import nltk
import os
import re
import random
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# 📦 Download NLTK data (ideally run this in setup instead of here)
nltk.download('punkt_tab')
nltk.download('stopwords')

# 🔄 Global models (load once)
whisper_model = whisper.load_model("tiny.en")  # Use 'base.en' for better accuracy if on GPU
summarizer = pipeline("summarization", model="t5-small", device=-1)  # Use device=0 if GPU available

# 🔽 Download YouTube video
def download_youtube_video(youtube_url, filename="youtube_video.mp4"):
    print(f"⬇️ Downloading YouTube video via yt-dlp: {youtube_url}")
    command = ["yt-dlp", "-f", "best[ext=mp4]+bestaudio/best", "-o", filename, youtube_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("YouTube download failed: " + result.stderr)
    return filename

# 🎧 Extract audio from video
def extract_audio(video_path):
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path

# 📝 Transcribe audio using Whisper
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# 📄 Generate summary in chunks
def generate_summary(text, default_max_len=130, default_min_len=30):
    sentences = sent_tokenize(text)
    chunks = [' '.join(sentences[i:i + 10]) for i in range(0, len(sentences), 10)]
    summary = ""

    for chunk in chunks:
        input_len = len(chunk.split())
        dynamic_max = max(20, min(default_max_len, input_len - 1))
        dynamic_min = max(10, min(default_min_len, dynamic_max - 10))

        result = summarizer(
            chunk,
            max_length=dynamic_max,
            min_length=dynamic_min,
            do_sample=False
        )[0]["summary_text"]

        summary += result + " "

    return summary.strip()

# ❓ Generate quiz
def generate_quiz(text, num_questions=5):
    sentences = sent_tokenize(text)
    tfidf = TfidfVectorizer(stop_words='english', max_features=300)
    X = tfidf.fit_transform(sentences)
    quiz = []
    used = set()

    for _ in range(num_questions):
        i = random.choice([x for x in range(len(sentences)) if x not in used])
        used.add(i)
        question = sentences[i]
        options = [question]

        while len(options) < 4:
            j = random.randint(0, len(sentences) - 1)
            if j != i and sentences[j] not in options:
                options.append(sentences[j])
        random.shuffle(options)
        quiz.append({
            "question": question,
            "options": options,
            "answer": question
        })

    return "\n\n".join([
        f"Q{i + 1}: {q['question']}\nOptions:\n" +
        "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(q['options'])])
        for i, q in enumerate(quiz)
    ])

# 📺 Subtitle formatting
def generate_subtitles(text, max_words_per_line=10):
    sentences = sent_tokenize(text)
    subtitles = []
    count = 1
    for sentence in sentences:
        chunks = [sentence[i:i + max_words_per_line] for i in range(0, len(sentence), max_words_per_line)]
        for chunk in chunks:
            subtitles.append(f"{count}. {chunk}")
            count += 1
    return "\n".join(subtitles)

# 🧪 Main processor
def process_video(video_path, selected_services):
    results = {}
    print("🔧 Extracting audio...")
    audio_path = extract_audio(video_path)

    if "Transcription" in selected_services:
        transcription = transcribe_audio(audio_path)
        results["transcription"] = transcription

        if "Summary" in selected_services:
            results["summary"] = generate_summary(transcription)

        if "Subtitles" in selected_services:
            results["subtitles"] = generate_subtitles(transcription)

        if "Quiz" in selected_services:
            results["quiz"] = generate_quiz(transcription)

    return results
