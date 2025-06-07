import gradio as gr
import shutil
from app import download_youtube_video, process_video

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
