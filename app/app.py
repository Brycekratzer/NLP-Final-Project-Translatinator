import os
import subprocess
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up device for Whisper
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Create pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

@app.route('/')
def index():
    """Serves the frontend page."""
    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handles audio upload and transcribes it using Whisper."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    webm_path = os.path.join(UPLOAD_FOLDER, "audio.webm")
    wav_path = os.path.join(UPLOAD_FOLDER, "audio.wav")

    audio_file.save(webm_path)

    # Convert WebM to WAV using FFmpeg
    try:
        subprocess.run(["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500

    # Transcribe with Whisper
    transcription = process_with_whisper(wav_path)

    return jsonify({"transcription": transcription})

def process_with_whisper(audio_path):
    """Runs Whisper transcription on the provided audio file."""
    try:
        result = pipe(audio_path, return_timestamps=True)
        return result["text"]
    except Exception as e:
        return f"Whisper processing error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
