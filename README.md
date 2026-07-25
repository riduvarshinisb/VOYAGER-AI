# Voyager AI

AI-personalized travel itinerary planner. Collects trip preferences, generates a day-wise itinerary, estimates a budget breakdown, shows trip-dated weather and a destination map, and exports everything as a PDF.

## Features

- Day-wise AI itinerary generation (Groq / Llama 3.1)
- Computed budget breakdown across 5 categories (₹)
- Trip-dated weather forecast (Open-Meteo, up to 16 days out)
- Destination map preview (Google Maps embed)
- Seasonal price trend outlook (LSTM)
- Real-data accommodation price check (NYC destinations)
- PDF export (itinerary + budget + weather + map link)
- Regenerate with feedback: select chips and/or type free-text notes to adjust the plan

## Tech stack

- Frontend: HTML / CSS / vanilla JS
- Backend: Node.js + Express
- AI: Groq API (Llama 3.1 8B)
- Weather: Open-Meteo (geocoding + forecast)
- Maps: Google Maps embed
- PDF: jsPDF (client-side)

## DL components

Three trained deep learning models, all loaded and run via hand-written forward passes directly in `server.js` — no separate Python service needed at runtime. Training code lives in `dl/`.

### 1. Feedback-adjustment model
`dl/train_bandit.py` → `bandit_weights.json`

A deep contextual bandit that predicts a budget-delta adjustment from user feedback.
- **Input (10 features):** 5 preset feedback chips + 5 keyword categories scored from free-text notes (shopping, nightlife, nature, luxury, family) via an LLM-based confidence score
- **Architecture:** 10 → 20 (ReLU) → 14 (ReLU) → 5 (linear output = predicted budget deltas)
- **Training:** epsilon-greedy exploration (1.0 → 0.05 over 250,000 episodes) + REINFORCE policy-gradient updates, against a simulated reward
- **Result:** ~0.90 worst-case / ~0.96 average reward across feature combinations

### 2. Seasonal price-trend forecaster
`dl/train_lstm.py` → `lstm_weights.json`

A hand-rolled LSTM (manual forward pass + backpropagation-through-time) that forecasts seasonal price trends by destination type.
- **Task:** given 14 days of a destination type's seasonal price-index history, predict the next day's index
- **Data:** simulated seasonal curves per destination type (beach, hill-station, city, heritage, desert) — stated plainly as simulated, not real historical pricing
- **Result:** test MAE of 0.028 (~2.8% typical error)

### 3. Real-data accommodation price model
`dl/train_price_model.py` + `dl/AB_NYC_2019.csv` → `price_model_weights.json`

A deep feedforward regression network trained on real Airbnb listing data.
- **Dataset:** "New York City Airbnb Open Data" by Dgomonov (Kaggle, 2019), compiled from Inside Airbnb (insideairbnb.com). Licensed CC BY 4.0. ~48,895 real listings.
- **Architecture:** 12 → 32 (ReLU) → 16 (ReLU) → 1, trained with a hand-implemented Adam optimizer
- **Result:** test MAE $52.32 vs. $72.47 for a naive mean-price baseline
- **Scope:** wired into the app only for NYC-matched destinations; not used for other destinations

## Setup

```bash
git clone https://github.com/riduvarshinisb/AI_TRAVEL_PLANNER.git
cd AI_TRAVEL_PLANNER
npm install
```

Create a `.env` file in the root:
```
GROQ_API_KEY=your_key_here
```

Run:
```bash
npm start          # production
npm run dev        # auto-restarts on server changes (nodemon)
```

Open `http://localhost:3000`.

## Retraining the DL models

```bash
cd dl
python3 train_bandit.py
python3 train_lstm.py
python3 train_price_model.py
```
Each overwrites its corresponding `*_weights.json` in the project root (requires `numpy`; the price model also requires `pandas`).

