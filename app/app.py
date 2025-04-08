import os
import subprocess
import torch
from flask import Flask, render_template, request, jsonify, send_file
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, MarianMTModel, MarianTokenizer, pipeline

# Used for file creation
import random
import string

# Used for searching directories
from pathlib import Path

# Used to compresss files
import zipfile

def generate_sequence(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ZIP_FOLDER = 'zipped'
TEMP_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ZIP_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

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

# Need to get recordings
@app.route('/collect', methods=['GET'])
def get_recordings():
    """Grab the recordings in the upload folder and send them back"""
    dir = os.getcwd()
    print(dir)

    zip_file = 'translations.zip'
    zip_filepath = os.path.join(ZIP_FOLDER, zip_file)
    
    # Remove file if already exists
    try:
        os.remove(zip_filepath)
    except Exception as e:
        print('*Fails silently*')
    
    files = []
    path = Path(UPLOAD_FOLDER)
    for child in path.iterdir():
        files.append(child.name)

    print(files)

    # Create the compressed file
    with zipfile.ZipFile(zip_filepath, "w") as zip:
        for filename in files:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            zip.write(file_path, arcname=filename)

    # os.chdir("..")
    return send_file(os.path.join(os.getcwd(), zip_filepath), as_attachment=True)

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

    webm_path = os.path.join(UPLOAD_FOLDER, f"audio_{sequence}.webm")
    wav_path = os.path.join(UPLOAD_FOLDER, f"audio_{sequence}.wav")

    transcription_save = os.path.join(UPLOAD_FOLDER, f"transcription_{sequence}.txt")

    # print(webm_path)
    # print(wav_path)

    audio_file.save(webm_path)

    # Convert WebM to WAV using FFmpeg
    try:
        subprocess.run(["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "16000", wav_path], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500

    # Transcribe with Whisper
    transcription = process_with_whisper(wav_path)

    # Write to file and save
    file = open(transcription_save, "w")
    file.write(transcription)
    file.close()

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

@app.route('/translate_eng_selected', methods=['POST'])
def translate_eng_selected():
    """Handles audio translation from a selected transcription."""
    
    # Gets global variable for transcription
    global transcription

    # Receive temporary text file
    temp_path = os.path.join(TEMP_FOLDER, f"temp.txt")

    if 'text' not in request.files:
        return jsonify({"error":"no text file provided!"}), 400

    text_file = request.files['text']

    text_file.save(temp_path)

    file = open(temp_path)
    transcription = file.read()
    file.close()

    # Gets translation from model
    result = translator_pipe_en_spa(transcription)

    # Returns the parsed translation from the model
    return jsonify({"translation": result[0]['translation_text']})

@app.route('/translate_spa_selected', methods=['POST'])
def translate_spa_selected():
    """Handles audio translation from a selected transcription."""
    
    # Gets global variable for transcription
    global transcription

    # Receive temporary text file
    temp_path = os.path.join(TEMP_FOLDER, f"temp.txt")

    if 'text' not in request.files:
        return jsonify({"error":"no text file provided!"}), 400

    text_file = request.files['text']

    text_file.save(temp_path)

    file = open(temp_path)
    transcription = file.read()
    file.close()

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
