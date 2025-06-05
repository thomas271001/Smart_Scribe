import whisper
from moviepy import VideoFileClip
from transformers import pipeline
import nltk
import os
import re
import random
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 0: Download YouTube video with yt-dlp
# -------------------------------
def download_youtube_video(youtube_url, filename="youtube_video.mp4"):
    print(f"⬇️ Downloading YouTube video via yt-dlp: {youtube_url}")
    command = ["yt-dlp", "-f", "best[ext=mp4]+bestaudio/best", "-o", filename, youtube_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading video:", result.stderr)
        raise Exception("YouTube download failed")
    return filename

# -------------------------------
# Step 1: Extract audio from video
# -------------------------------
def extract_audio(video_path, audio_path="temp_audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

# -------------------------------
# Step 2: Transcribe audio using Whisper
# -------------------------------
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

# -------------------------------
# Step 3: Correct transcription using LLM
# -------------------------------
grammar_corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

def correct_text_with_llm(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sent in sentences:
        if len(sent.strip()) > 1:
            try:
                corrected = grammar_corrector(sent, max_length=128, do_sample=False)[0]['generated_text']
                corrected_sentences.append(corrected)
            except Exception as e:
                print(f"Error correcting sentence: {sent}\n{e}")
                corrected_sentences.append(sent)
    return ' '.join(corrected_sentences)

# -------------------------------
# Step 4: Create subtitles in .srt format
# -------------------------------
def generate_srt(transcription_result, srt_path="subtitles.srt"):
    segments = transcription_result['segments']
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            start = seg['start']
            end = seg['end']
            text = seg['text']
            f.write(f"{i+1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text.strip()}\n\n")
    return srt_path

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02}:{m:02}:{int(s):02},{ms:03}"

# -------------------------------
# Step 5: Summarize the transcript
# -------------------------------
def summarize_text(text, max_length=150):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# -------------------------------
# Step 6: Get context-aware distractors
# -------------------------------
def get_distractors(correct_answer, text, num_distractors=3):
    words = list(set(re.findall(r'\b\w+\b', text.lower())))
    words = [w for w in words if w != correct_answer.lower() and w not in stop_words and len(w) > 3 and w.isalpha()]

    if not words:
        return ["OptionA", "OptionB", "OptionC"][:num_distractors]

    vectorizer = TfidfVectorizer().fit_transform([correct_answer] + words)
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    sorted_indices = similarity.argsort()
    distractors = [words[i] for i in sorted_indices[:num_distractors]]

    return distractors

# -------------------------------
# Step 7: Generate MCQs
# -------------------------------
mcq_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def generate_mcqs(text, num_questions=5):
    sentences = sent_tokenize(text)
    mcqs = []
    count = 0

    for sent in sentences:
        if count >= num_questions:
            break
        try:
            words = sent.split()
            if len(words) < 6:
                continue
            answer = words[-1].strip('.,')
            if answer.lower() in ['is', 'are', 'the', 'a', 'an', 'of']:
                continue
            highlight = sent.replace(answer, f"<hl> {answer} <hl>")
            prompt = f"generate question: {highlight}"

            result = mcq_generator(prompt, max_length=128, do_sample=False)
            question = result[0]['generated_text']

            distractors = get_distractors(answer, text)
            options = [answer] + distractors
            random.shuffle(options)

            mcqs.append({
                "question": question,
                "options": options,
                "answer": answer
            })
            count += 1
        except Exception as e:
            print(f"Skipping due to error: {e}")
    return mcqs

# -------------------------------
# Step 8: Extract bullet point notes
# -------------------------------
def extract_notes_with_llm(text, max_points=7):
    note_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = note_summarizer(text, max_length=200, min_length=60, do_sample=False)[0]['summary_text']
    bullet_points = sent_tokenize(summary)
    return bullet_points[:max_points]

# -------------------------------
# Main pipeline
# -------------------------------
def process_video(video_path_or_url):
    if video_path_or_url.startswith("http"):
        video_path = download_youtube_video(video_path_or_url)
    else:
        video_path = video_path_or_url

    print("📅 Extracting audio...")
    audio_file = extract_audio(video_path)

    print("🖍️ Transcribing audio...")
    transcription_result = transcribe_audio(audio_file)
    raw_transcript = transcription_result["text"]

    print("🤖 Correcting transcription with LLM...")
    corrected_transcript = correct_text_with_llm(raw_transcript)

    print("🎬 Generating subtitles...")
    generate_srt(transcription_result)

    print("🧠 Summarizing transcript...")
    summary = summarize_text(corrected_transcript)

    print("❓ Generating MCQs...")
    mcqs = generate_mcqs(corrected_transcript)

    print("📝 Extracting notes...")
    notes = extract_notes_with_llm(corrected_transcript)

    return {
        "raw_transcript": raw_transcript,
        "corrected_transcript": corrected_transcript,
        "summary": summary,
        "mcqs": mcqs,
        "notes": notes
    }

# -------------------------------
# Run script example
# -------------------------------
if __name__ == "__main__":
    input_source = "https://www.youtube.com/watch?v=ttIOdAdQaUE"
    output = process_video(input_source)

    print("\n📜 Raw Transcription:\n", output["raw_transcript"])
    print("\n✅ Corrected Transcription:\n", output["corrected_transcript"])
    print("\n🔍 Summary:\n", output["summary"])

    if output["mcqs"]:
        print("\n❓ MCQ Questions:")
        for i, mcq in enumerate(output["mcqs"], 1):
            print(f"Q{i}: {mcq['question']}")
            for opt in mcq["options"]:
                print(f"- {opt}")
            print(f"Answer: {mcq['answer']}\n")
    else:
        print("\n❌ No valid MCQs generated.")

    print("\n📌 Notes:")
    for note in output["notes"]:
        print("•", note)
