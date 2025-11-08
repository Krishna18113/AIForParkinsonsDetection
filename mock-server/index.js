import express from "express";
import multer from "multer";
import cors from "cors";

const app = express();
const upload = multer();

app.use(cors());

// Mock /predict endpoint
app.post("/predict", upload.single("audio"), (req, res) => {
  // Simulate random Parkinson's risk between 0 and 1
  const randomRisk = Math.random();
  console.log("🎧 Received audio file, sending mock risk:", randomRisk);
  res.json({ risk_score: randomRisk });
});

// Start server
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`✅ Mock server running at http://localhost:${PORT}`);
});
