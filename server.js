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

// --- Seasonal price-trend forecaster (LSTM, trained offline) ---
const lstmModel = JSON.parse(fs.readFileSync(path.join(__dirname, "lstm_weights.json"), "utf-8"));

function detectDestType(destination) {
  const lower = destination.toLowerCase();
  for (const [type, info] of Object.entries(lstmModel.dest_types)) {
    if (info.keywords.some((kw) => lower.includes(kw))) return type;
  }
  return "city"; // safest default (lowest seasonal swing)
}

function dayOfYear(dateStr) {
  const d = new Date(dateStr);
  const start = new Date(d.getFullYear(), 0, 0);
  return Math.floor((d - start) / 86400000);
}

function seasonalValue(doy, amplitude, peakDay) {
  return 1 + amplitude * Math.cos((2 * Math.PI * (doy - peakDay)) / 365);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
}

function lstmForward(inputSeq) {
  const { hidden, weights: W } = lstmModel;
  let h = new Array(hidden).fill(0);
  let c = new Array(hidden).fill(0);

  for (const x of inputSeq) {
    const z = [...h, ...x]; // concat hidden + input

    const gate = (Wname, bname) =>
      W[bname].map((_, j) => {
        let sum = W[bname][j];
        for (let k = 0; k < z.length; k++) sum += z[k] * W[Wname][k][j];
        return sum;
      });

    const f = gate("Wf", "bf").map(sigmoid);
    const i = gate("Wi", "bi").map(sigmoid);
    const o = gate("Wo", "bo").map(sigmoid);
    const g = gate("Wg", "bg").map(Math.tanh);

    c = c.map((cv, idx) => f[idx] * cv + i[idx] * g[idx]);
    h = c.map((cv, idx) => o[idx] * Math.tanh(cv));
  }

  let y = W.by[0];
  for (let k = 0; k < hidden; k++) y += h[k] * W.Wy[k][0];
  return y;
}

function getSeasonalOutlook(destination, startDate) {
  const type = detectDestType(destination);
  const { amplitude, peak_day } = lstmModel.dest_types[type];
  const startDoy = dayOfYear(startDate);

  const seq = [];
  for (let d = startDoy - lstmModel.seq_len; d < startDoy; d++) {
    const doy = ((d % 365) + 365) % 365;
    const val = seasonalValue(doy, amplitude, peak_day);
    seq.push([val, Math.sin((2 * Math.PI * doy) / 365), Math.cos((2 * Math.PI * doy) / 365)]);
  }

  const predicted = lstmForward(seq);
  const percentDiff = Math.round((predicted - 1) * 100);

  const typeLabel = type.charAt(0).toUpperCase() + type.slice(1);
  let message;
  if (percentDiff >= 12) {
    message = `Peak season for this type of destination — prices trending ~${percentDiff}% above average.`;
  } else if (percentDiff <= -8) {
    message = `Off-season — prices trending ~${Math.abs(percentDiff)}% below average, good time to visit for savings.`;
  } else {
    message = `Prices trending close to the annual average (~${percentDiff >= 0 ? "+" : ""}${percentDiff}%) for your dates.`;
  }

  return { type: typeLabel, percentDiff, message };
}

// --- Real-data accommodation price model (trained on NYC Airbnb data) ---
const priceModel = JSON.parse(fs.readFileSync(path.join(__dirname, "price_model_weights.json"), "utf-8"));
const NYC_KEYWORDS = ["new york", "nyc", "manhattan", "brooklyn"];

function isNYC(destination) {
  const lower = destination.toLowerCase();
  return NYC_KEYWORDS.some((kw) => lower.includes(kw));
}

function predictNightlyPrice(borough, roomType) {
  const { boroughs, room_types, num_mean, num_std, weights: W } = priceModel;

  const rawNumeric = [Math.log1p(3), Math.log1p(15), 200, 1.0];
  const numeric = rawNumeric.map((v, i) => (v - num_mean[i]) / num_std[i]);
  const boroughVec = boroughs.map((b) => (b === borough ? 1 : 0));
  const roomVec = room_types.map((r) => (r === roomType ? 1 : 0));
  const x = [...numeric, ...boroughVec, ...roomVec];

  const h1 = W.W1[0].map((_, j) => {
    let s = W.b1[j];
    for (let i = 0; i < x.length; i++) s += x[i] * W.W1[i][j];
    return Math.max(0, s);
  });
  const h2 = W.W2[0].map((_, j) => {
    let s = W.b2[j];
    for (let i = 0; i < h1.length; i++) s += h1[i] * W.W2[i][j];
    return Math.max(0, s);
  });
  let out = W.b3[0];
  for (let i = 0; i < h2.length; i++) out += h2[i] * W.W3[i][0];

  return Math.round(Math.expm1(out));
}

function getRealDataPriceInsight(destination) {
  if (!isNYC(destination)) return null;
  const nightly = predictNightlyPrice("Manhattan", "Entire home/apt");
  return {
    nightlyUsd: nightly,
    message: `Real-data check: comparable Manhattan entire-home Airbnb listings average ~$${nightly}/night (model trained on 48,645 real NYC listings, test MAE $${priceModel.test_mae_usd.toFixed(0)}).`,
  };
}

// --- Feedback-adjustment model (trained offline, loaded here for inference) ---
const bandit = JSON.parse(
  fs.readFileSync(path.join(__dirname, "bandit_weights.json"), "utf-8")
);

async function scoreKeywords(freeText) {
  const zeroScores = Object.fromEntries(bandit.keyword_features.map((k) => [k, 0]));
  if (!freeText || !freeText.trim()) return zeroScores;

  try {
    const response = await axios.post(
      GROQ_URL,
      {
        model: "llama-3.1-8b-instant",
        messages: [
          {
            role: "system",
            content:
              "You rate travel feedback text against categories. Respond with ONLY a valid JSON object, no other text.",
          },
          {
            role: "user",
            content: `Rate how strongly this feedback expresses interest in each category, from 0.0 (not at all) to 1.0 (very strongly): ${bandit.keyword_features.join(", ")}.\nFeedback: "${freeText}"\nRespond as JSON only, e.g. {"shopping":0.0,"nightlife":0.0,"nature":0.0,"luxury":0.0,"family":0.0}`,
          },
        ],
      },
      { headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` } }
    );

    const raw = response.data?.choices?.[0]?.message?.content || "{}";
    const parsed = JSON.parse(raw.match(/\{[\s\S]*\}/)?.[0] || "{}");

    const scores = { ...zeroScores };
    for (const k of bandit.keyword_features) {
      const v = Number(parsed[k]);
      scores[k] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
    }
    return scores;
  } catch (error) {
    console.error("Keyword scoring failed, defaulting to no signal:", error.message);
    return zeroScores;
  }
}

async function predictAdjustment(selectedChips, freeText) {
  const activeChips = bandit.chip_features.filter((f) =>
    selectedChips.includes(f.replace(/_/g, "-"))
  );
  const keywordScores = await scoreKeywords(freeText);
  const hasKeywordSignal = Object.values(keywordScores).some((v) => v > 0.15);

  if (activeChips.length === 0 && !hasKeywordSignal) {
    return null;
  }

  const context = bandit.features.map((f) =>
    bandit.chip_features.includes(f) ? (activeChips.includes(f) ? 1 : 0) : keywordScores[f]
  );
  const { W1, b1, W2, b2, W3, b3 } = bandit;

  const hidden1 = W1[0].map((_, j) => {
    let sum = b1[j];
    for (let i = 0; i < context.length; i++) sum += context[i] * W1[i][j];
    return Math.max(0, sum);
  });

  const hidden2 = W2[0].map((_, j) => {
    let sum = b2[j];
    for (let i = 0; i < hidden1.length; i++) sum += hidden1[i] * W2[i][j];
    return Math.max(0, sum);
  });

  const deltas = W3[0].map((_, j) => {
    let sum = b3[j];
    for (let i = 0; i < hidden2.length; i++) sum += hidden2[i] * W3[i][j];
    return sum;
  });

  const instruction = activeChips.map((f) => bandit.chip_instructions[f]).join("; ");
  const strongKeywords = bandit.keyword_features.filter((f) => keywordScores[f] > 0.4);
  const labelParts = [
    ...activeChips.map((f) => f.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())),
    ...strongKeywords.map((f) => `${f.replace(/\b\w/g, (c) => c.toUpperCase())} (from notes)`),
  ];

  return { deltas, instruction, label: labelParts.join(", ") };
}

// API endpoint for itinerary generation
app.post("/generate", async (req, res) => {
  const { destination, budget, days, preferences, feedback, feedbackChips, startDate } = req.body;

  if (!destination || !budget || !days) {
    return res.status(400).json({ error: "Please fill in all fields." });
  }

  const adjustment = feedback || (feedbackChips && feedbackChips.length)
    ? await predictAdjustment(feedbackChips || [], feedback)
    : null;

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
            content: `Plan a ${days}-day trip to ${destination} with a budget of ₹${budget} (Indian Rupees). Include top attractions, meal options, accommodation, and travel tips, with all costs quoted in Indian Rupees (₹). Preferences: ${preferences}.${feedback ? ` The user reviewed a previous version of this plan and gave this feedback — adjust accordingly: ${feedback}.` : ""}${adjustment && adjustment.instruction ? ` Specifically, ${adjustment.instruction}.` : ""} Do not include your own overall "Budget Breakdown" or "Total Budget" summary section — a separate, authoritative budget breakdown is already shown to the user elsewhere in the app. Only mention individual prices naturally within the day-by-day plan where relevant.`,
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
      seasonalOutlook: startDate ? getSeasonalOutlook(destination, startDate) : null,
      realDataPriceInsight: getRealDataPriceInsight(destination),
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

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tripStart = new Date(startDate);
    tripStart.setHours(0, 0, 0, 0);
    const offsetDays = Math.round((tripStart - today) / 86400000);
    const tripDays = Number(days);

    const MAX_FORECAST_DAYS = 16;

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