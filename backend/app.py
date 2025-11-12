import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import soundfile as sf
import io
import warnings
# We will need librosa for the feature extraction
# import librosa 

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

app = Flask(__name__)
# Enable CORS to allow your React app (running on localhost:3000 or similar)
# to communicate with this API (running on localhost:5000)
CORS(app) 

# --- 1. Load the Trained Model ---
MODEL_PATH = 'parkinsons_rf_classifier.pkl'
classifier = None

try:
    # This script expects the .pkl file to be in the same folder
    with open(MODEL_PATH, 'rb') as f:
        classifier = pickle.load(f)
    print(f"Model '{MODEL_PATH}' loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_PATH}' not found.")
    print("Please make sure 'parkinsons_rf_classifier.pkl' is in the same directory as app.py")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 2. Define Feature Order (CRITICAL) ---
# This MUST match the order from your training script
FEATURE_ORDER = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# --- 3. Audio Feature Extraction Function (OUR NEXT TASK) ---
def extract_features_from_audio(audio_data, sample_rate):
    """
    This is the core of the hackathon MVP.
    It takes raw audio data (as a NumPy array) and must return the 22 features.
    
    This function is currently a PLACEHOLDER.
    We will implement the real logic here next.
    """
    print(f"Received audio data for feature extraction, sample rate: {sample_rate}")
    
    try:
        # ---
        # HACKATHON TO-DO: Implement real feature extraction here
        #
        # Example (this is NOT the full 22 features, just a concept):
        # f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # mean_f0 = np.nanmean(f0[voiced_flag])
        #
        # ... and so on for all 22 features ...
        # ---
        
        # --- MOCKUP LOGIC (To be replaced) ---
        # For now, we return mock data to test the API pipeline
        # This mock data is based on an average "Parkinson's" patient (status=1)
        mock_features = [
            145.188, 179.916, 104.99, 0.0076, 4e-05, 0.003, 0.004, 0.009, 
            0.040, 0.380, 0.020, 0.027, 0.030, 0.060, 0.020, 20.0, 
            0.500, 0.700, -5.0, 0.250, 2.300, 0.300
        ]
        
        print("MOCKUP: Returning 22 mock features.")
        return mock_features
    
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

# --- 4. Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if classifier is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500
        
    try:
        # 1. Get the audio file from the FormData (matching your React code's key: "audio")
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file found in request. Make sure the key is "audio".'}), 400

        audio_file = request.files['audio']
        
        # 2. Read the audio file in memory
        # We use soundfile to read the audio blob (wav, webm, etc.)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))
        
        # If the audio is stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        print(f"Successfully read audio file. Sample Rate: {sample_rate}, Duration: {len(audio_data) / sample_rate:.2f}s")

        # 3. Extract features (Currently using mock data)
        input_features = extract_features_from_audio(audio_data, sample_rate)
        
        if input_features is None or len(input_features) != 22:
            return jsonify({'error': f'Feature extraction failed. Expected 22 features, got {len(input_features) if input_features else 0}.'}), 400

        # 4. Format data for the model
        input_data = np.array(input_features).reshape(1, -1)
        
        # 5. Make Prediction
        prediction_index = classifier.predict(input_data)[0]
        prediction_proba = classifier.predict_proba(input_data)[0]
        
        # The probability of class 1 (Parkinson's)
        # This is what your frontend code expects as `risk_score`
        risk_score_prob_parkinsons = float(prediction_proba[1]) 
        
        result_label = "Parkinson's Risk Detected" if prediction_index == 1 else "Healthy"
        
        print(f"Prediction: {result_label}, Risk Score: {risk_score_prob_parkinsons:.4f}")
        
        # 6. Return result to React frontend (matching your App.jsx)
        return jsonify({
            'risk_score': risk_score_prob_parkinsons,
            'prediction_label': result_label,
            'status': 'success'
        })

    except Exception as e:
        print(f"[ERROR] Prediction endpoint failed: {e}")
        # Send a user-friendly error to your React app
        if "soundfile.LibsndfileError" in str(e):
             return jsonify({'error': 'Failed to process audio file. Please try recording in .wav or .webm format.'}), 400
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

# --- 5. Main execution ---
if __name__ == '__main__':
    # Run the Flask app
    print("Starting Flask ML Prediction API...")
    print("Listening on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)