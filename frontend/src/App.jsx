import { useState } from "react";
import { ReactMediaRecorder } from "react-media-recorder";
import axios from "axios";
import "./index.css";

function App() {
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleStop = async (blobUrl) => {
    const blob = await fetch(blobUrl).then((r) => r.blob());
    const formData = new FormData();
    formData.append("audio", blob, "voice.wav");

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setRisk(res.data.risk_score);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-8">
      <h1 className="text-3xl font-bold mb-6">🧠 Parkinson’s Voice Test</h1>

      <ReactMediaRecorder
        audio
        onStop={handleStop}
        render={({ status, startRecording, stopRecording, mediaBlobUrl }) => (
          <div className="flex flex-col items-center">
            <p className="mb-4 text-lg font-medium">Status: {status}</p>

            <div className="flex gap-4 mb-4">
              <button
                onClick={startRecording}
                className="px-4 py-2 bg-green-600 text-white rounded-lg"
              >
                🎙 Start Recording
              </button>
              <button
                onClick={stopRecording}
                className="px-4 py-2 bg-red-600 text-white rounded-lg"
              >
                ⏹ Stop
              </button>
            </div>

            {mediaBlobUrl && (
              <audio src={mediaBlobUrl} controls className="mb-4"></audio>
            )}

            {loading && <p>Analyzing...</p>}
            {risk && !loading && (
              <p className="text-xl">
                Parkinson’s Risk Score:{" "}
                <span className="font-bold">{risk.toFixed(2)}</span>
              </p>
            )}
          </div>
        )}
      />
    </div>
  );
}

export default App;
