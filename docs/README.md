# The Translate-Inator

The Translate-Inator is a web application that provides speech-to-text transcription and translation capabilities between English and Spanish. Built with Flask, PyTorch, and Hugging Face Transformers, this application allows users to record audio, transcribe it, and translate the text between English and Spanish languages in real-time.

## Features

- **Speech-to-Text Transcription**: Record audio through your microphone and convert it to text using OpenAI's Whisper large-v3-turbo model
- **Bidirectional Translation**: Translate text between English and Spanish using MarianMT neural translation models
- **Recording History**: View and access previous recordings and their transcriptions
- **Responsive UI**: Clean, user-friendly interface for all operations

## Technology Stack

### Backend
- **Flask**: Web framework for the application
- **PyTorch**: Deep learning framework powering the ML models
- **Transformers**: Hugging Face's library for state-of-the-art NLP models
- **Whisper**: OpenAI's automatic speech recognition model
- **MarianMT**: Neural machine translation model for text translation

### Frontend
- **HTML/CSS/JavaScript**: Basic frontend implementation
- **JSZip**: Library for handling ZIP files in JavaScript

## Architecture

The application is structured with a Flask backend that serves the frontend and handles API requests for transcription and translation. The main components include:

1. **Speech Recognition Module**: Uses Whisper large-v3-turbo model to transcribe audio recordings
2. **Translation Module**: Uses MarianMT models for bidirectional translation between English and Spanish
3. **History Management**: Stores recordings and transcriptions for future reference

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install flask torch transformers ffmpeg-python
   ```
3. Make sure FFmpeg is installed on your system for audio conversion
4. Run the application:
   ```
   python app/app.py
   ```
5. Access the web interface at `http://localhost:5000`

## Usage

1. Click "Start Recording" to begin capturing audio from your microphone
2. Speak into the microphone
3. Click "Stop Recording" when finished
4. The transcription will appear in the transcription box
5. Click "Translate to Spanish" or "Translate to English" to translate the text
6. View previous recordings in the history panel and select them to work with previous transcriptions

## Project Structure

- **app/**: Contains the main application code
  - **app.py**: Main Flask application with API endpoints and ML model setup
  - **static/**: Frontend assets (JavaScript, CSS)
  - **templates/**: HTML templates
  - **uploads/**: Storage for audio recordings and transcriptions
- **docs/**: Documentation for models and examples
- **models/**: Code for fine-tuning and evaluating the MarianMT models

## Model Information

### Whisper Speech Recognition

The application uses OpenAI's Whisper large-v3-turbo model for speech recognition. This model is state-of-the-art in multilingual speech recognition and provides high-quality transcriptions for both English and Spanish audio.

### MarianMT Translation

Translation is performed using pre-trained MarianMT models:
- **English to Spanish**: Helsinki-NLP/opus-mt-en-es
- **Spanish to English**: Helsinki-NLP/opus-mt-es-en

The repository also includes code for fine-tuning these models, although the evaluation (in `models/Evaluate.ipynb`) shows that the fine-tuned models perform very similarly to the base models.

## Research and Development

The project includes notebooks and scripts for:
- Fine-tuning MarianMT models on customized datasets
- Evaluating model performance using BERTScore
- Running training jobs on GPU clusters using SLURM

## Future Improvements

Potential areas for enhancement:
- Adding support for more languages
- Improving the UI/UX
- Implementing real-time translation
- Adding pronunciation features
- Expanding the history management system

## Contributors

This project was developed as part of an NLP course project.

## License