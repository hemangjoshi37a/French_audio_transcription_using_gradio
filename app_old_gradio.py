# Using Flask to create a REST API that can be deployed on VM.
# This will start a Flask server on your VM that listens on port 8080. You can then send a POST request to this server with an audio file attached as the `file` parameter, and it will return the transcription as a JSON object.

from flask import Flask, jsonify, request
from transformers import pipeline
import torchaudio
from waitress import serve
app = Flask(__name__)

# pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-large-v2-french")
pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-medium-french")

pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")

@app.route('/transcribe', methods=['POST'],timeout=None)
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
    
    return jsonify({'transcription': generated_sentences})

if __name__ == '__main__':
    # serve(app, host="0.0.0.0", port=8080)
    app.run(host='0.0.0.0', port=8081, debug=True)