import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();
const app = express();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Serve frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

// Environment variables
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

// API endpoint for itinerary generation
app.post("/generate", async (req, res) => {
  const { destination, budget, days, preferences } = req.body;

  if (!destination || !budget || !days) {
    return res.status(400).json({ error: "Please fill in all fields." });
  }

  try {
    const response = await axios.post(
      GROQ_URL,
      {
        model: "llama-3.1-8b-instant", 
        messages: [
          {
            role: "system",
            content:
              "You are a professional travel planner. Generate detailed, clearly structured itineraries without markdown tables or emojis. Use day-wise headers and clean formatting.",
          },
          {
            role: "user",
            content: `Plan a ${days}-day trip to ${destination} with a budget of ${budget} USD. Include top attractions, meal options, accommodation, and travel tips. Preferences: ${preferences}.`,
          },
        ],
        temperature: 0.8,
      },
      {
        headers: {
          Authorization: `Bearer ${GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    const itinerary = response.data?.choices?.[0]?.message?.content || "No response generated.";
    res.json({ itinerary });
  } catch (error) {
    console.error("Error from Groq API:", error.response?.data || error.message);
    res.status(500).json({
      error: error.response?.data || "Something went wrong. Please try again later.",
    });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`Server running at http://localhost:${PORT}`)
);

