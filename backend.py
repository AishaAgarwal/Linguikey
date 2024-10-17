from flask import Flask, request, jsonify, render_template
import os
import wave
from vosk import Model, KaldiRecognizer
import json
from rake_nltk import Rake

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(file_path):
    model = Model("vosk-model-small-en-us-0.15")  # Path to your Vosk model
    wf = wave.open(file_path, "rb")
    
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        raise ValueError("Audio format is not supported, please upload a single-channel, 16-bit, 8kHz/16kHz WAV file")

    rec = KaldiRecognizer(model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            if 'text' in result_dict:
                text += result_dict['text'] + " "
    wf.close()

    keywords = extract_keywords_rake(text)
    return text, keywords

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords_with_scores = rake.get_ranked_phrases_with_scores()
    return [kw for score, kw in keywords_with_scores]  # Return only the keywords

@app.route('/')
def upload_form():
    return render_template('index.html')  

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        text, keywords = transcribe_audio(file_path)
        return jsonify({'transcription': text.strip(), 'keywords': keywords})

    return jsonify({'error': 'File type not allowed'})

if __name__ == '__main__':
    app.run(debug=True)
