# Voyager AI

AI-personalized travel itinerary planner. Collects trip preferences, generates a day-wise itinerary, estimates a budget breakdown, shows trip-dated weather and a destination map, and exports everything as a PDF.

## Features

- Day-wise AI itinerary generation (Groq / Llama 3.1)
- Computed budget breakdown across 5 categories (₹)
- Trip-dated weather forecast (Open-Meteo, up to 16 days out)
- Destination map preview (Google Maps embed)
- PDF export (itinerary + budget + weather + map link)
- Regenerate with feedback: select chips and/or type free-text notes to adjust the plan

## Tech stack

- Frontend: HTML / CSS / vanilla JS
- Backend: Node.js + Express
- AI: Groq API (Llama 3.1 8B)
- Weather: Open-Meteo (geocoding + forecast)
- Maps: Google Maps embed
- PDF: jsPDF (client-side)
- Deployment: Render

## DL component: feedback-adjustment model

`dl/train_bandit.py` trains a deep contextual bandit that predicts a budget-delta adjustment from user feedback.

- **Input (10 features):** 5 preset feedback chips + 5 keyword categories detected from free-text notes (shopping, nightlife, nature, luxury, family)
- **Architecture:** 10 → 20 (ReLU) → 14 (ReLU) → 5 (linear output = predicted budget deltas)
- **Training:** epsilon-greedy exploration (1.0 → 0.05 over 250,000 episodes) + REINFORCE policy-gradient updates, against a simulated reward (1 − normalized distance to an ideal blended target)
- **Result:** ~0.90 worst-case / ~0.96 average reward across feature combinations
- **Deployment:** trained weights exported to `bandit_weights.json`, loaded and run via a hand-written forward pass directly in `server.js` — no separate Python service needed at runtime

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

## Retraining the feedback model

```bash
cd ml
python3 train_bandit.py
```
Overwrites `bandit_weights.json` in the project root (requires `numpy`).

