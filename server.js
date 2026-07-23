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
            content: `Plan a ${days}-day trip to ${destination} with a budget of ₹${budget} (Indian Rupees). Include top attractions, meal options, accommodation, and travel tips, with all costs quoted in Indian Rupees (₹). Preferences: ${preferences}.`,
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

