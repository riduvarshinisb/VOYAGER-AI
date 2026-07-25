import express from "express";
import axios from "axios";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

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

// WMO weather codes → plain descriptions
const WEATHER_CODES = {
  0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
  45: "Foggy", 48: "Foggy",
  51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
  61: "Light rain", 63: "Rain", 65: "Heavy rain",
  71: "Light snow", 73: "Snow", 75: "Heavy snow",
  80: "Rain showers", 81: "Rain showers", 82: "Violent rain showers",
  95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
};
function describeWeather(code) {
  return WEATHER_CODES[code] || "Weather unavailable";
}

// --- Feedback-adjustment model (trained offline, loaded here for inference) ---
// v2: predicts a continuous, BLENDED budget-delta vector directly, instead
// of picking one of several fixed "arms" - this avoids the tie-break bug
// where two valid feedback signals (e.g. "more culture" + "relax the pace")
// would silently cancel down to only one winner being acted on.
const bandit = JSON.parse(
  fs.readFileSync(path.join(__dirname, "bandit_weights.json"), "utf-8")
);

function predictAdjustment(selectedChips) {
  const activeFeatures = bandit.features.filter((f) =>
    selectedChips.includes(f.replace(/_/g, "-"))
  );

  if (activeFeatures.length === 0) {
    return null; // no feedback signal to act on
  }

  const context = bandit.features.map((f) => (activeFeatures.includes(f) ? 1 : 0));
  const { W1, b1, W2, b2 } = bandit;

  // Layer 1: context -> hidden, with ReLU
  const hidden = W1[0].map((_, j) => {
    let sum = b1[j];
    for (let i = 0; i < context.length; i++) sum += context[i] * W1[i][j];
    return Math.max(0, sum);
  });

  // Layer 2: hidden -> predicted budget deltas (continuous, one per category)
  const deltas = W2[0].map((_, j) => {
    let sum = b2[j];
    for (let i = 0; i < hidden.length; i++) sum += hidden[i] * W2[i][j];
    return sum;
  });

  // Instruction + label are built deterministically from EVERY active chip
  // (not chosen by the network), so no selected concern is ever dropped.
  const instruction = activeFeatures.map((f) => bandit.chip_instructions[f]).join("; ");
  const label = activeFeatures
    .map((f) => f.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()))
    .join(", ");

  return { deltas, instruction, label };
}

// API endpoint for itinerary generation
app.post("/generate", async (req, res) => {
  const { destination, budget, days, preferences, feedback, feedbackChips } = req.body;

  const adjustment = feedbackChips && feedbackChips.length
    ? predictAdjustment(feedbackChips)
    : null;

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
            content: `Plan a ${days}-day trip to ${destination} with a budget of ₹${budget} (Indian Rupees). Include top attractions, meal options, accommodation, and travel tips, with all costs quoted in Indian Rupees (₹). Preferences: ${preferences}.${feedback ? ` The user reviewed a previous version of this plan and gave this feedback — adjust accordingly: ${feedback}.` : ""}${adjustment && adjustment.instruction ? ` Specifically, ${adjustment.instruction}.` : ""}`,
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
    res.json({
      itinerary,
      budgetAdjustment: adjustment
        ? { label: adjustment.label, deltas: adjustment.deltas }
        : null,
    });
  } catch (error) {
    console.error("Error from Groq API:", error.response?.data || error.message);
    res.status(500).json({
      error: error.response?.data || "Something went wrong. Please try again later.",
    });
  }
});

// API endpoint for weather forecast
app.post("/weather", async (req, res) => {
  const { destination, startDate, days } = req.body;

  if (!destination || !startDate || !days) {
    return res.status(400).json({ error: "Destination, start date, and days are required." });
  }

  try {
    const geoRes = await axios.get("https://geocoding-api.open-meteo.com/v1/search", {
      params: { name: destination, count: 1 },
    });

    const place = geoRes.data?.results?.[0];
    if (!place) {
      return res.status(404).json({ error: "Location not found for weather lookup." });
    }

    // How many days from today does the trip start?
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tripStart = new Date(startDate);
    tripStart.setHours(0, 0, 0, 0);
    const offsetDays = Math.round((tripStart - today) / 86400000);
    const tripDays = Number(days);

    const MAX_FORECAST_DAYS = 16; // limit of free forecast data

    // Trip starts too far in the future for any forecast to be meaningful
    if (offsetDays >= MAX_FORECAST_DAYS) {
      return res.json({
        location: `${place.name}, ${place.country}`,
        outOfRange: true,
      });
    }

    const clampedOffset = Math.max(offsetDays, 0);
    const forecastDaysNeeded = Math.min(MAX_FORECAST_DAYS, clampedOffset + tripDays);

    const weatherRes = await axios.get("https://api.open-meteo.com/v1/forecast", {
      params: {
        latitude: place.latitude,
        longitude: place.longitude,
        daily: "temperature_2m_max,temperature_2m_min,weather_code",
        timezone: "auto",
        forecast_days: forecastDaysNeeded,
      },
    });

    const { daily } = weatherRes.data;
    const tripForecast = daily.time
      .map((date, i) => ({
        date,
        max: daily.temperature_2m_max[i],
        min: daily.temperature_2m_min[i],
        condition: describeWeather(daily.weather_code[i]),
      }))
      .slice(clampedOffset);

    res.json({
      location: `${place.name}, ${place.country}`,
      outOfRange: false,
      partial: tripForecast.length < tripDays,
      forecast: tripForecast,
    });
  } catch (error) {
    console.error("Error from Weather API:", error.response?.data || error.message);
    res.status(500).json({ error: "Could not fetch weather data." });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`Server running at http://localhost:${PORT}`)
);

