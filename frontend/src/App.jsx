// App.jsx (replace your current file content with this)
import { useState } from "react";
import { ReactMediaRecorder } from "react-media-recorder";
import axios from "axios";
import "./index.css";

function blobToWavBlob(blob) {
  // Convert arbitrary audio blob (e.g., webm) to WAV using WebAudio API
  // Returns a Promise that resolves to a WAV Blob (PCM 16-bit, mono)
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        // convert to mono float32 array
        let channelData = audioBuffer.numberOfChannels > 1
          ? audioBuffer.getChannelData(0).map((v, i) => {
              // average channels if stereo, small overhead
              let sum = v;
              for (let c = 1; c < audioBuffer.numberOfChannels; c++) {
                sum += audioBuffer.getChannelData(c)[i];
              }
              return sum / audioBuffer.numberOfChannels;
            })
          : audioBuffer.getChannelData(0);

        // ensure Float32Array
        channelData = Float32Array.from(channelData);

        // encode to 16-bit PCM wav
        const sampleRate = audioBuffer.sampleRate;
        const buffer = new ArrayBuffer(44 + channelData.length * 2);
        const view = new DataView(buffer);

        /* WAV header */
        function writeString(view, offset, string) {
          for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
          }
        }
        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + channelData.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, channelData.length * 2, true);

        // PCM 16-bit
        let offset = 44;
        for (let i = 0; i < channelData.length; i++, offset += 2) {
          let s = Math.max(-1, Math.min(1, channelData[i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }

        const wavBlob = new Blob([view], { type: "audio/wav" });
        resolve(wavBlob);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = () => reject(new Error("Failed to read blob"));
    reader.readAsArrayBuffer(blob);
  });
}

function App() {
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleStop = async (blobUrl) => {
    try {
      setLoading(true);
      const blob = await fetch(blobUrl).then((r) => r.blob());
      // Convert to WAV before upload (makes backend simpler)
      const wavBlob = await blobToWavBlob(blob);

      const formData = new FormData();
      formData.append("audio", wavBlob, "voice.wav");

      const res = await axios.post(
        "http://localhost:5000/predict",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 60_000,
        }
      );
      setRisk(res.data.risk_score);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend: " + (err?.response?.data?.error || err.message));
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

            {mediaBlobUrl && <audio src={mediaBlobUrl} controls className="mb-4"></audio>}

            {loading && <p>Analyzing...</p>}
            {risk && !loading && (
              <p className="text-xl">
                Parkinson’s Safe Score: <span className="font-bold">{risk.toFixed(2)}</span>
              </p>
            )}
          </div>
        )}
      />
    </div>
  );
}

export default App;
