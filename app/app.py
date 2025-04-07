import os
import subprocess
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, MarianMTModel, MarianTokenizer, pipeline

# Used for file creation
import random
import string

def generate_sequence(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

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

# Load MarianMT model
model_en_spa = "Helsinki-NLP/opus-mt-en-es"
translator_model_en_spa = MarianMTModel.from_pretrained(model_en_spa)

# Load MarianMT Tokenizor
translator_tokenizer = MarianTokenizer.from_pretrained(model_en_spa)

# Create Translation Pipeline
translator_pipe_en_spa = pipeline(
    "translation",
    model=translator_model_en_spa,
    tokenizer=translator_tokenizer,
    device=device  # Reuse the same device as Whisper
)

# Load MarianMT model
model_spa_en = "Helsinki-NLP/opus-mt-es-en"
translator_model_spa_en = MarianMTModel.from_pretrained(model_spa_en)

# Load MarianMT Tokenizor
translator_tokenizer = MarianTokenizer.from_pretrained(model_spa_en)

# Create Translation Pipeline
translator_pipe_spa_en = pipeline(
    "translation",
    model=translator_model_spa_en,
    tokenizer=translator_tokenizer,
    device=device  # Reuse the same device as Whisper
)

# Global variable for translation access
transcription = ""

@app.route('/')
def index():
    """Serves the frontend page."""
    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handles audio upload and transcribes it using Whisper."""
    
    # Allows for reference to global variable 
    global transcription
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # Generate a 6-character sequence
    sequence = generate_sequence()
    # print(sequence) 

    webm_path = os.path.join(UPLOAD_FOLDER, f"audio{sequence}.webm")
    wav_path = os.path.join(UPLOAD_FOLDER, f"audio{sequence}.wav")

    print(webm_path)
    print(wav_path)

    audio_file.save(webm_path)

    # Convert WebM to WAV using FFmpeg
    try:
        subprocess.run(["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500

    # Transcribe with Whisper
    transcription = process_with_whisper(wav_path)

    return jsonify({"transcription": transcription})

@app.route('/en_spa_trans', methods=['POST'])
def translation_en_spa():
    """Handles audio translation from transcription."""
    
    # Gets global variable for transcription
    global transcription
    
    # Gets translation from model
    result = translator_pipe_en_spa(transcription)
    
    # Returns the parsed translation from the model
    return jsonify({"translation": result[0]['translation_text']})

@app.route('/spa_en_trans', methods=['POST'])
def translation_spa_en():
    """Handles audio translation from transcription."""
    
    # Gets global variable for transcription
    global transcription
    
    # Gets translation from model
    result = translator_pipe_spa_en(transcription)
    
    # Returns the parsed translation from the model
    return jsonify({"translation": result[0]['translation_text']})

def process_with_whisper(audio_path):
    """Runs Whisper transcription on the provided audio file."""
    try:
        result = pipe(audio_path, return_timestamps=True)
        return result["text"]
    except Exception as e:
        return f"Whisper processing error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
