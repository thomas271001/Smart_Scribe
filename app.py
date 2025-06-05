import whisper
from moviepy import VideoFileClip
from transformers import pipeline
import nltk
import re
import os

# Download NLTK punkt tokenizer
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

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
corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

def correct_text_with_llm(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []
    for sent in sentences:
        # Ensure sentence is not too short or empty
        if len(sent.strip()) > 1:
            try:
                corrected = corrector(sent, max_length=128, do_sample=False)[0]['generated_text']
                corrected_sentences.append(corrected)
            except Exception as e:
                print(f"Error correcting sentence: {sent}\n{e}")
                corrected_sentences.append(sent)  # fallback
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
    return f"{h:02}:{m:02}:{int(s):02},{int((s - int(s)) * 1000):03}"

# -------------------------------
# Step 5: Summarize the transcript
# -------------------------------
def summarize_text(text, max_length=150):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# -------------------------------
# Step 6: Generate fill-in-the-blank quiz questions
# -------------------------------
def generate_quiz(text, num_questions=5):
    sentences = sent_tokenize(text)
    questions = []
    for i in range(min(num_questions, len(sentences))):
        sentence = sentences[i]
        question = re.sub(r'\b(is|are|was|were|has|have|had|will|can|did)\b', '____', sentence, 1, flags=re.IGNORECASE)
        questions.append((f"Q{i+1}: {question}", sentence))
    return questions

# -------------------------------
# Step 7: Extract bullet point notes
# -------------------------------
def extract_notes(text, max_bullets=7):
    sentences = sent_tokenize(text)
    return sentences[:max_bullets]

# -------------------------------
# Main pipeline
# -------------------------------
def process_video(video_path):
    print("📥 Extracting audio...")
    audio_file = extract_audio(video_path)

    print("📝 Transcribing audio...")
    transcription_result = transcribe_audio(audio_file)
    raw_transcript = transcription_result["text"]

    print("🤖 Correcting transcription with LLM...")
    corrected_transcript = correct_text_with_llm(raw_transcript)

    print("🎬 Generating subtitles...")
    generate_srt(transcription_result)

    print("🧠 Summarizing transcript...")
    summary = summarize_text(corrected_transcript)

    print("🧾 Creating quiz questions...")
    quiz = generate_quiz(corrected_transcript)

    print("🗒️ Extracting notes...")
    notes = extract_notes(corrected_transcript)

    return {
        "raw_transcript": raw_transcript,
        "corrected_transcript": corrected_transcript,
        "summary": summary,
        "quiz": quiz,
        "notes": notes
    }

# -------------------------------
# Script Usage
# -------------------------------
if __name__ == "__main__":
    video_path = "/content/How To Use AI To Make Faceless Videos In Less Than 5 Minutes.mp4"  # Replace with your actual file path
    output = process_video(video_path)

    print("\n📜 Raw Transcription:\n", output["raw_transcript"])
    print("\n✅ Corrected Transcription:\n", output["corrected_transcript"])
    print("\n🔍 Summary:\n", output["summary"])

    print("\n❓ Quiz Questions:")
    for q, a in output["quiz"]:
        print(f"{q}\nAnswer: {a}\n")

    print("\n📌 Notes:")
    for note in output["notes"]:
        print("•", note)
