import whisper
from moviepy.editor import VideoFileClip
from transformers import pipeline
import nltk
import os
import re
import subprocess
import torch
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Initialize pipelines globally to avoid reloading
print("🔄 Loading models...")
grammar_corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 0: Download YouTube video
def download_youtube_video(youtube_url, filename="youtube_video.mp4"):
    print(f"⬇️ Downloading YouTube video: {youtube_url}")
    command = ["yt-dlp", "-f", "best[ext=mp4]+bestaudio/best", "-o", filename, youtube_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("YouTube download failed")
    return filename

# Step 1: Extract audio
def extract_audio(video_path, audio_path="temp_audio.wav"):
    print("🎵 Extracting audio from video...")
    video = VideoFileClip(video_path)
    if video.audio is None:
        raise Exception("No audio stream found")
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    return audio_path

# Step 2: Transcribe audio
def transcribe_audio(audio_path):
    print("🎙️ Loading Whisper model and transcribing audio...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

# Step 3: Correct transcription
def correct_text_with_llm(text):
    print("🧹 Correcting grammar and text...")
    sentences = sent_tokenize(text)
    corrected = []

    for i, sent in enumerate(sentences):
        print(f"Correcting sentence {i+1}/{len(sentences)}")
        try:
            output = grammar_corrector(sent, max_length=128, do_sample=False)[0]['generated_text']
            corrected.append(output)
        except Exception as e:
            print(f"⚠️ Failed to correct sentence: {str(e)}")
            corrected.append(sent)

    return ' '.join(corrected)

# Step 4: Generate SRT subtitles
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def generate_srt(transcription_result, srt_path="subtitles.srt"):
    print("🎬 Generating SRT subtitles...")
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(transcription_result['segments']):
            f.write(f"{i+1}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text'].strip()}\n\n")
    return srt_path

# Step 5: Summarize
def summarize_text(text, max_length=150):
    print("🔍 Generating summary...")
    try:
        summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"⚠️ Summarization failed: {str(e)}")
        # Fallback to first few sentences
        sentences = sent_tokenize(text)
        return ' '.join(sentences[:3])

# Step 6: Improved Quiz generation
def clean_and_validate_question(question):
    """
    Clean and validate generated questions
    """
    if not question:
        return None

    # Remove common prefixes
    question = re.sub(r'^(question:|q:|generate question:|answer:)\s*', '', question, flags=re.IGNORECASE)

    # Clean up the question
    question = question.strip()

    # Ensure it ends with a question mark
    if not question.endswith('?'):
        question += '?'

    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:]

    # Validation criteria
    validation_checks = [
        len(question) > 10,                                    # Minimum length
        len(question) < 200,                                   # Maximum length
        question.count('?') == 1,                              # Only one question mark
        question.endswith('?'),                                # Ends with question mark
        not question.lower().startswith(('generate', 'create', 'write', 'make')),  # Not a command
        any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']),  # Contains question words
        not re.search(r'\b(example|instance|sample)\b', question.lower()),  # Not asking for examples
        len([w for w in question.split() if w.isalpha()]) >= 4  # At least 4 actual words
    ]

    if all(validation_checks):
        return question
    else:
        return None

def generate_simple_questions(text, num_needed):
    """
    Generate simple questions using template-based approach as fallback
    """
    sentences = sent_tokenize(text)
    questions = []

    # Simple question templates
    templates = [
        "What is mentioned about {}?",
        "How does {} work?",
        "Why is {} important?",
        "When does {} occur?",
        "Where can {} be found?"
    ]

    # Extract key nouns and phrases
    try:
        for sent in sentences[:10]:  # Limit to first 10 sentences
            if len(questions) >= num_needed:
                break

            # Extract nouns using simple regex for key terms
            words = re.findall(r'\b[A-Z][a-z]+\b', sent)  # Capitalized words
            nouns = [word.lower() for word in words if len(word) > 3]

            for noun in nouns[:2]:  # Max 2 nouns per sentence
                if len(questions) >= num_needed:
                    break

                template = templates[len(questions) % len(templates)]
                question = template.format(noun)

                questions.append({
                    'question': question,
                    'source': sent,
                    'context': sent
                })

    except Exception as e:
        print(f"⚠️ Error in fallback question generation: {str(e)}")

    return questions

def generate_quiz_questions(text, num_questions=5):
    """
    Generate quiz questions from text using improved logic and validation
    """
    print(f"❓ Generating {num_questions} quiz questions...")

    try:
        # Initialize question generation pipeline
        qg_pipeline = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
    except Exception as e:
        print(f"⚠️ Failed to load primary QG model: {str(e)}")
        print("🔄 Falling back to template-based questions...")
        return generate_simple_questions(text, num_questions)

    sentences = sent_tokenize(text)
    questions = []
    seen_questions = set()

    # Filter sentences - keep only substantial ones
    substantial_sentences = [
        sent.strip() for sent in sentences
        if len(sent.strip()) > 20 and len(sent.strip()) < 300
    ]

    print(f"Found {len(substantial_sentences)} substantial sentences to process")

    for i, sent in enumerate(substantial_sentences):
        if len(questions) >= num_questions:
            break

        print(f"Processing sentence {i+1}/{min(len(substantial_sentences), num_questions*2)}")

        try:
            # Clean the sentence
            clean_sent = re.sub(r'[^\w\s.,!?-]', '', sent).strip()

            # Generate question using proper format for the model
            input_text = f"context: {clean_sent} </s>"

            result = qg_pipeline(
                input_text,
                max_length=64,
                min_length=10,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )

            generated_question = result[0]['generated_text'].strip()

            # Clean and validate the question
            question = clean_and_validate_question(generated_question)

            if question and question not in seen_questions:
                questions.append({
                    'question': question,
                    'source': sent.strip(),
                    'context': clean_sent
                })
                seen_questions.add(question)
                print(f"✅ Generated: {question}")
            else:
                print(f"❌ Rejected: {generated_question}")

        except Exception as e:
            print(f"⚠️ Error generating question for sentence: {str(e)}")
            continue

    # If we don't have enough questions, try alternative approach
    if len(questions) < num_questions:
        print(f"🔄 Only generated {len(questions)} questions, trying fallback approach...")
        additional_questions = generate_simple_questions(text, num_questions - len(questions))
        questions.extend(additional_questions)

    print(f"✅ Successfully generated {len(questions)} questions")
    return questions[:num_questions]

# Step 7: Notes extraction
def extract_notes(text, max_points=7):
    print("📒 Extracting key notes...")
    try:
        summary = summarize_text(text, max_length=200)
        notes = sent_tokenize(summary)[:max_points]
        return notes
    except Exception as e:
        print(f"⚠️ Note extraction failed: {str(e)}")
        # Fallback to first few sentences
        sentences = sent_tokenize(text)
        return sentences[:max_points]

# Main pipeline
def process_video(video_path_or_url):
    """
    Main function to process video and extract all information
    """
    print("🚀 Starting video processing pipeline...")

    # Step 0: Handle video source
    if video_path_or_url.startswith("http"):
        print("🌐 Processing YouTube URL...")
        video_path = download_youtube_video(video_path_or_url)
    else:
        print("📁 Processing local video file...")
        video_path = video_path_or_url
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Step 1: Extract audio
        print("📅 Extracting audio...")
        audio_path = extract_audio(video_path)

        # Step 2: Transcribe
        print("📝 Transcribing...")
        transcription = transcribe_audio(audio_path)
        raw_text = transcription['text']

        if not raw_text.strip():
            raise Exception("Transcription resulted in empty text")

        # Step 3: Correct grammar
        print("🧹 Correcting grammar...")
        corrected_text = correct_text_with_llm(raw_text)

        # Step 4: Generate subtitles
        print("🎬 Generating subtitles...")
        srt_path = generate_srt(transcription)

        # Step 5: Summarize
        print("🔍 Summarizing...")
        summary = summarize_text(corrected_text)

        # Step 6: Create questions
        print("❓ Creating questions...")
        questions = generate_quiz_questions(corrected_text)

        # Step 7: Extract notes
        print("📒 Extracting notes...")
        notes = extract_notes(corrected_text)

        # Clean up temporary files
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print("🗑️ Cleaned up temporary audio file")
        except Exception as e:
            print(f"⚠️ Could not clean up audio file: {str(e)}")

        print("✅ Video processing completed successfully!")

        return {
            "raw_transcript": raw_text,
            "corrected_transcript": corrected_text,
            "srt_path": srt_path,
            "summary": summary,
            "questions": questions,
            "notes": notes,
            "video_path": video_path
        }

    except Exception as e:
        print(f"❌ Error in video processing: {str(e)}")
        raise

# Enhanced display function
def display_results(output):
    """
    Display results in a formatted way
    """
    print("\n" + "="*80)
    print("📋 VIDEO PROCESSING RESULTS")
    print("="*80)

    print(f"\n📜 RAW TRANSCRIPTION ({len(output['raw_transcript'])} characters):")
    print("-" * 50)
    print(output['raw_transcript'][:500] + "..." if len(output['raw_transcript']) > 500 else output['raw_transcript'])

    print(f"\n✅ CORRECTED TRANSCRIPTION ({len(output['corrected_transcript'])} characters):")
    print("-" * 50)
    print(output['corrected_transcript'][:500] + "..." if len(output['corrected_transcript']) > 500 else output['corrected_transcript'])

    print(f"\n🔍 SUMMARY:")
    print("-" * 50)
    print(output['summary'])

    if output['questions']:
        print(f"\n❓ QUIZ QUESTIONS ({len(output['questions'])} questions):")
        print("-" * 50)
        for i, q in enumerate(output['questions'], 1):
            if isinstance(q, dict):
                print(f"Q{i}: {q['question']}")
            else:
                print(f"Q{i}: {q}")
    else:
        print("\n❌ No questions generated.")

    print(f"\n📌 KEY NOTES ({len(output['notes'])} points):")
    print("-" * 50)
    for i, note in enumerate(output['notes'], 1):
        print(f"{i}. {note}")

    print(f"\n🎬 FILES GENERATED:")
    print("-" * 50)
    print(f"• Subtitles: {output['srt_path']}")
    if 'video_path' in output:
        print(f"• Video: {output['video_path']}")

    print("\n" + "="*80)

# Example usage and main execution
if __name__ == "__main__":
    # Example usage - modify this path/URL as needed
    input_source = "/content/videoplayback.mp4"  # Change this to your video path or YouTube URL

    try:
        print("🎬 Starting video processing...")
        output = process_video(input_source)
        display_results(output)
        print("🎉 Processing completed successfully!")

    except FileNotFoundError as e:
        print(f"❌ File not found: {str(e)}")
        print("💡 Please check the video path or URL")

    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        print("💡 Please check your video file and try again")


# Step 6: Generate quiz (simple placeholder version)
def generate_quiz(text):
    # Simple demo logic — replace with LLM-based generation if needed
    return "1. What is the main topic discussed?\n2. List two key takeaways from the video."

# Main video processing function
def process_video(video_path, selected_services):
    results = {}

    # Step 1: Extract audio
    audio_path = extract_audio(video_path)

    # Step 2: Transcribe
    transcription_result = transcribe_audio(audio_path)
    raw_text = transcription_result["text"]

    if "Transcription" in selected_services:
        corrected_text = correct_text_with_llm(raw_text)
        results["transcription"] = corrected_text
    else:
        corrected_text = raw_text

    # Step 3: Subtitles
    if "Subtitles" in selected_services:
        srt_file = generate_srt(transcription_result)
        with open(srt_file, "r", encoding="utf-8") as f:
            results["subtitles"] = f.read()

    # Step 4: Summary
    if "Summary" in selected_services:
        summary = summarize_text(corrected_text)
        results["summary"] = summary

    # Step 5: Quiz
    if "Quiz" in selected_services:
        results["quiz"] = generate_quiz(corrected_text)

    return results

import gradio as gr
import shutil

def smartscribe_interface(video_file, youtube_link, services):
    if not video_file and not youtube_link:
        return "❌ Please upload a video or paste a YouTube link.", None, None, None, None

    input_path = "input_video.mp4"
    if youtube_link:
        download_youtube_video(youtube_link, filename=input_path)
    else:
        shutil.copy(video_file, input_path)

    results = process_video(input_path, services)

    return (
        results.get("transcription", "N/A"),
        results.get("summary", "N/A"),
        results.get("subtitles", "N/A"),
        results.get("quiz", "N/A"),
    )

with gr.Blocks() as demo:
    gr.Markdown("# 🎓 SmartScribe - AI-Powered Learning Assistant")

    with gr.Row():
        video_input = gr.Video(label="📤 Upload a Video File (MP4)")
        youtube_input = gr.Textbox(label="📎 Or Paste a YouTube Link")

    services = gr.CheckboxGroup(
        ["Transcription", "Summary", "Subtitles", "Quiz"],
        label="🛠️ Select Services"
    )

    submit_btn = gr.Button("🚀 Process Video")

    transcription_output = gr.Textbox(label="📄 Transcription")
    summary_output = gr.Textbox(label="📝 Summary")
    subtitle_output = gr.Textbox(label="🎬 Subtitles (SRT or Text)")
    quiz_output = gr.Textbox(label="❓ Auto-Generated Quiz")

    submit_btn.click(
        smartscribe_interface,
        inputs=[video_input, youtube_input, services],
        outputs=[transcription_output, summary_output, subtitle_output, quiz_output]
    )

demo.launch()
