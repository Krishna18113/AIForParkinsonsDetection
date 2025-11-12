# 🧠 AI for Early Detection of Parkinson’s from Voice

A **Minimal Viable Product (MVP)** application designed for **early detection of Parkinson’s Disease (PD)** risk using **pre-extracted acoustic features** from voice recordings.

This repository contains the complete full-stack project for the hackathon submission:

- **`backend/`** → Python Flask API that loads a pre-trained scikit-learn model, performs audio feature extraction, and serves predictions via a REST endpoint.  
- **`frontend/`** → React (Vite) application that records the user’s voice, sends it to the backend, and displays the PD risk score.

---

## 🏗️ Architecture Overview

The application follows a standard **service-oriented architecture**:

1. The **Frontend** uses native browser APIs (`MediaRecorder`) to capture raw audio.  
2. The recorded audio is sent via a **`POST`** request to the **Backend API**.  
3. The **Backend** performs **feature extraction** (using `librosa`, etc.) to convert the audio into 22 biomedical features used for training.  
4. These features are passed into a **pre-trained Random Forest Classifier** (`parkinsons_rf_classifier.pkl`).  
5. The **risk score (probability of PD)** is returned to the frontend.

---

## 🚀 Quick Start — Run the MVP Locally

To run the application locally, ensure you have **Python 3.x**, **pip**, and **npm** installed.

### 🧩 1. Start the Backend (Python API)

This assumes you already have the trained model file  
`backend/parkinsons_rf_classifier.pkl`.

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # (On Windows: .\.venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```
### 💻 2. Start the Frontend (React UI)

The frontend automatically connects to the Flask API (http://localhost:5000).
```bash
cd frontend
npm install
npm run dev

Then visit the URL shown in your terminal (usually http://localhost:5173).
```

## 📡 API Contract — POST /predict
The backend exposes a single endpoint for prediction:
| Detail              | Specification                   |
| :------------------ | :------------------------------ |
| **Endpoint**        | `POST /predict`                 |
| **Data Format**     | `multipart/form-data`           |
| **File Field Name** | `audio` *(must match this key)* |
| **Audio Format**    | WAV (mono recommended)          |

## 🧩 Tech Stack
| Layer                  | Technology                              |
| :--------------------- | :-------------------------------------- |
| **Frontend**           | React + Vite + MediaRecorder API        |
| **Backend**            | Python + Flask                          |
| **ML Model**           | Scikit-Learn (Random Forest Classifier) |
| **Feature Extraction** | Librosa + NumPy                         |
| **Communication**      | REST (JSON over HTTP)                   |

## 🏁 Summary

This MVP demonstrates how AI and voice analysis can assist in early detection of Parkinson’s Disease by leveraging audio biomarkers and machine learning.

---

# Start the Flask API server
python app.py
