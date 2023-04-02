from flask import Flask, request, render_template,jsonify
import gradio as gr
import numpy as np
from transformers import pipeline
import torchaudio
from waitress import serve
app = Flask(__name__)


# pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-large-v2-french")
pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-medium-french")

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file found'})
    
    audio_file = request.files['file']
    waveform, sample_rate = torchaudio.load(audio_file)
    
    required_sample_rate = 16000  
    if sample_rate != required_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=required_sample_rate)
        waveform = resampler(waveform)
    
    waveform_np = waveform.squeeze().numpy()
    generated_sentences = pipe(waveform_np, max_new_tokens=1000)["text"]
    
    # return jsonify({'transcription': generated_sentences})
    return generated_sentences


# Run the app
if __name__ == '__main__':
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600
    app.config['RESPONSE_TIMEOUT'] = 0
    app.run(debug=True, host='0.0.0.0', port=8080)
    # serve(app, host="0.0.0.0", port=8080,cleanup_interval=999999999,channel_timeout=999999)
    