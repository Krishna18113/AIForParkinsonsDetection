# app.py (replace your existing file with this)
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import soundfile as sf
import io
import warnings
import librosa
import aubio
import tempfile
import traceback

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = 'parkinsons_rf_classifier.pkl'
classifier = None

try:
    with open(MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    print(f"Model '{MODEL_PATH}' loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_PATH}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")

FEATURE_ORDER = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

def extract_features_from_audio(audio_data, sample_rate):
    TARGET_SR = 22050
    if sample_rate != TARGET_SR:
        audio_data = librosa.resample(audio_data.astype(float), orig_sr=sample_rate, target_sr=TARGET_SR)
        sample_rate = TARGET_SR

    win_s = 1024
    hop_s = 256
    pitch_o = aubio.pitch("default", win_s, hop_s, sample_rate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.8)

    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    f0_values = []
    # process audio in hops of hop_s; aubio.pitch expects frames of length hop_s
    for i in range(0, len(audio_data), hop_s):
        frame = audio_data[i:i + hop_s]
        # pad last frame to hop_s with zeros if needed
        if len(frame) < hop_s:
            pad = np.zeros(hop_s - len(frame), dtype=frame.dtype)
            frame = np.concatenate((frame, pad))
        # aubio requires float32
        pitch_val = pitch_o(frame.astype(np.float32))[0]
        if pitch_val > 0:
            f0_values.append(pitch_val)


    f0_voiced = np.array(f0_values)
    if len(f0_voiced) < 5:
        print("Warning: Too few voiced frames detected. Cannot extract features.")
        return None

    f0_mean = float(np.mean(f0_voiced))
    f0_max = float(np.max(f0_voiced))
    f0_min = float(np.min(f0_voiced))

    # Mock/average values (same as your previous values)
    Jitter_per = 0.00622; Jitter_abs = 0.00004; RAP = 0.0033; PPQ = 0.00344; DDP = 0.0099
    Shimmer_local = 0.034; Shimmer_dB = 0.300; APQ3 = 0.0176; APQ5 = 0.0202; APQ_MDVP = 0.0284; DDA = 0.0526
    NHR = 0.0248; HNR = 20.06
    RPDE = 0.522; DFA = 0.760; spread1 = -5.684; spread2 = 0.228; D2 = 2.408; PPE = 0.218

    input_features = [
        f0_mean, f0_max, f0_min,
        Jitter_per, Jitter_abs, RAP, PPQ, DDP,
        Shimmer_local, Shimmer_dB, APQ3, APQ5, APQ_MDVP, DDA,
        NHR, HNR,
        RPDE, DFA, spread1, spread2, D2, PPE
    ]

    return input_features

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if classifier is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    try:
        # Accept file under key 'audio' (FormData upload)
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file found in request. Make sure the key is "audio".'}), 400

        audio_file = request.files['audio']
        print("Received file:", getattr(audio_file, 'filename', None), "mimetype:", audio_file.mimetype)

        # Save to temp file with extension based on filename or default .wav
        filename = getattr(audio_file, 'filename', None) or "upload.wav"
        ext = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(audio_file.read())

        # Try reading with soundfile first (works for WAV)
        try:
            audio_data, sample_rate = sf.read(tmp_path)
        except Exception as e_sf:
            # fallback: use librosa (audioread backend) which can open more formats if ffmpeg available
            try:
                audio_data, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
            except Exception as e_lib:
                # Clean up and re-raise a friendly error
                os.remove(tmp_path)
                traceback.print_exc()
                return jsonify({'error': 'Failed to process audio file. Please send a WAV file.'}), 400

        # Ensure mono numpy array
        if hasattr(audio_data, 'ndim') and audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        duration_sec = len(audio_data) / float(sample_rate)
        print(f"Read audio: sr={sample_rate}, duration={duration_sec:.2f}s")

        MAX_DURATION_SECONDS = 10.0
        if duration_sec > MAX_DURATION_SECONDS:
            audio_data = audio_data[:int(MAX_DURATION_SECONDS * sample_rate)]

        # Extract features
        input_features = extract_features_from_audio(audio_data, sample_rate)
        if input_features is None or len(input_features) != 22:
            # clean up
            try: os.remove(tmp_path)
            except: pass
            got = len(input_features) if input_features else 0
            return jsonify({'error': f'Feature extraction failed. Expected 22 features, got {got}.'}), 400

        input_data = np.array(input_features).reshape(1, -1)
        prediction_index = classifier.predict(input_data)[0]
        prediction_proba = classifier.predict_proba(input_data)[0]
        risk_score_prob_parkinsons = float(prediction_proba[1])
        result_label = "Parkinson's Risk Detected" if prediction_index == 1 else "Healthy"

        print(f"Prediction: {result_label}, score={risk_score_prob_parkinsons:.4f}")
        try: os.remove(tmp_path)
        except: pass

        return jsonify({
            'risk_score': risk_score_prob_parkinsons,
            'prediction_label': result_label,
            'status': 'success'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask ML Prediction API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
