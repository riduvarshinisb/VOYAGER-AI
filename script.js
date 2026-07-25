document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("trip-form");
  const outputDiv = document.getElementById("output");
  const itineraryText = document.getElementById("itinerary-text");
  const downloadBtn = document.getElementById("download-pdf");
  const feedbackPanel = document.getElementById("feedback-panel");
  const feedbackChips = document.querySelectorAll(".chip");
  const feedbackText = document.getElementById("feedback-text");
  const regenerateBtn = document.getElementById("regenerate-btn");
  let lastFormData = null;

  feedbackChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chip.classList.toggle("selected");
    });
  });
  const budgetBreakdownDiv = document.getElementById("budget-breakdown");
  const weatherInfoDiv = document.getElementById("weather-info");
  const mapContainerDiv = document.getElementById("map-container");
  let lastWeather = null;
  let lastBudgetDeltas = null;

  function getBudgetBreakdown(totalBudget, deltas = null) {
    const base = [0.35, 0.25, 0.15, 0.15, 0.10];
    const labels = ["Accommodation", "Food & Dining", "Local Transport", "Activities & Sightseeing", "Shopping & Misc."];

    let percents = base;
    if (deltas && deltas.length === 5) {
      percents = base.map((p, i) => Math.max(0.03, p + deltas[i] / 100));
      const sum = percents.reduce((a, b) => a + b, 0);
      percents = percents.map((p) => p / sum); // renormalize to 100%
    }

    return labels.map((label, i) => ({
      label,
      amount: totalBudget * percents[i],
    }));
  }

  function renderBudgetBreakdown(totalBudget) {
    const breakdown = getBudgetBreakdown(totalBudget);
    const rows = breakdown
      .map(
        (item) =>
          `<div class="budget-row"><span>${item.label}</span><span>₹${Math.round(item.amount).toLocaleString("en-IN")}</span></div>`
      )
      .join("");
    return `<div class="budget-row budget-total"><span>Total Budget</span><span>₹${Math.round(totalBudget).toLocaleString("en-IN")}</span></div>${rows}`;
  }

  function renderWeather(weather) {
    if (!weather) return "";

    if (weather.outOfRange) {
      return `<p class="weather-note">Weather forecast isn't available yet for dates this far out — check back closer to your trip (within about 2 weeks) for a live forecast.</p>`;
    }

    if (!weather.forecast || weather.forecast.length === 0) return "";

    const days = weather.forecast
      .map((d) => {
        const label = new Date(d.date).toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" });
        return `<div class="weather-day">
          <div class="day-label">${label}</div>
          <div class="day-temp">${Math.round(d.max)}° / ${Math.round(d.min)}°</div>
        </div>`;
      })
      .join("");

    const partialNote = weather.partial
      ? `<p class="weather-note">Showing forecast for the first ${weather.forecast.length} day(s) of your trip — the rest are too far out to forecast yet.</p>`
      : "";

    return `<div class="weather-forecast">${days}</div>${partialNote}`;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const destination = document.getElementById("destination").value.trim();
    const budget = document.getElementById("budget").value.trim();
    const days = document.getElementById("days").value.trim();
    const startDate = document.getElementById("start-date").value;
    const preferences = document.getElementById("preferences").value.trim();

    lastFormData = { destination, budget, days, startDate, preferences };

    // Show loading
    outputDiv.classList.remove("hidden");
    itineraryText.innerHTML = `<p>Generating your itinerary…</p>`;
    downloadBtn.classList.add("hidden");

    if (!destination || !budget || !days) {
      itineraryText.innerHTML = `<p style="color:red;">Please fill in all required fields.</p>`;
      return;
    }

    try {
      const [response, weatherResponse] = await Promise.all([
        fetch("/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ destination, budget, days, preferences }),
        }),
        fetch("/weather", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ destination, startDate, days }),
        }),
      ]);

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();

      if (weatherResponse.ok) {
        const weatherData = await weatherResponse.json();
        lastWeather = weatherData;
        weatherInfoDiv.innerHTML = renderWeather(weatherData);
      } else {
        lastWeather = null;
        weatherInfoDiv.innerHTML = "";
      }

      budgetBreakdownDiv.innerHTML = renderBudgetBreakdown(Number(budget), null);
      lastBudgetDeltas = null;

      const mapQuery = encodeURIComponent(destination);
      mapContainerDiv.innerHTML = `<iframe src="https://maps.google.com/maps?q=${mapQuery}&output=embed" loading="lazy" allowfullscreen></iframe>`;

      let text = data.itinerary || "No itinerary generated.";

      // 🧹 Clean and format for browser display
      text = text
        .replace(/\*\*/g, "")
        .replace(/#+/g, "<br><br><strong>")
        .replace(/\n{2,}/g, "<br><br>")
        .replace(/\n/g, "<br>")
        .replace(/- /g, "• ")
        .replace(/\|/g, " | ")
        .replace(/---/g, "<hr>")
        .replace(/> /g, "")
        .replace(/\*/g, "")
        .trim();

      itineraryText.innerHTML = `<div class="itinerary-box">${text}</div>`;
      downloadBtn.classList.remove("hidden");
      feedbackPanel.classList.remove("hidden");

      //PDF Generator
      downloadBtn.onclick = () => {
        const jsPDF = window.jspdf.jsPDF;
        const doc = new jsPDF({
          orientation: "portrait",
          unit: "pt",
          format: "a4",
        });

        // --- HEADER SECTION ---
        doc.setFillColor(34, 150, 243);
        doc.rect(0, 0, 595, 60, "F"); // blue banner
        doc.setFont("helvetica", "bold");
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(20);
        doc.text("Voyager AI", 40, 40);

        doc.setTextColor(0, 0, 0);
        doc.setFont("times", "bold");
        doc.setFontSize(16);
        doc.text(`Destination: ${destination}`, 40, 90);
        doc.setFont("times", "normal");
        doc.setFontSize(12);
        doc.text(`Budget: Rs. ${budget}  |  Duration: ${days} days`, 40, 110);
        doc.text(`Preferences: ${preferences || "N/A"}`, 40, 130);
        doc.setTextColor(30, 77, 91);
        doc.textWithLink("View destination on Google Maps", 40, 148, {
          url: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(destination)}`,
        });
        doc.setTextColor(0, 0, 0);

        const pageHeight = 842;
        const margin = 40;
        let y = 172;

        // --- WEATHER ---
        if (lastWeather && lastWeather.outOfRange) {
          doc.setFont("times", "italic");
          doc.setFontSize(10);
          doc.setTextColor(100, 100, 100);
          doc.text("Weather forecast not yet available this far in advance.", margin, y);
          y += 22;
        } else if (lastWeather && lastWeather.forecast && lastWeather.forecast.length) {
          doc.setFont("times", "bold");
          doc.setFontSize(13);
          doc.setTextColor(30, 77, 91);
          doc.text("Weather Forecast", margin, y);
          y += 18;

          doc.setFont("times", "normal");
          doc.setFontSize(11);
          doc.setTextColor(0, 0, 0);
          const forecastLine = lastWeather.forecast
            .map((d) => {
              const label = new Date(d.date).toLocaleDateString("en-IN", { weekday: "short" });
              return `${label}: ${Math.round(d.max)}/${Math.round(d.min)}°`;
            })
            .join("   ");
          doc.text(forecastLine, margin, y);
          y += 14;
          doc.setDrawColor(200, 200, 200);
          doc.line(margin, y, 550, y);
          y += 20;
        }

        // --- BUDGET BREAKDOWN ---
        doc.setFont("times", "bold");
        doc.setFontSize(13);
        doc.setTextColor(30, 77, 91);
        doc.text("Budget Breakdown", margin, y);
        y += 18;

        const breakdown = getBudgetBreakdown(Number(budget), lastBudgetDeltas);
        doc.setFontSize(11);
        breakdown.forEach((item) => {
          doc.setFont("times", "normal");
          doc.setTextColor(0, 0, 0);
          doc.text(item.label, margin, y);
          doc.text(`Rs. ${Math.round(item.amount).toLocaleString("en-IN")}`, 550, y, { align: "right" });
          y += 16;
        });
        y += 14;
        doc.setDrawColor(200, 200, 200);
        doc.line(margin, y, 550, y);
        y += 20;

        // --- CONTENT BODY ---

        const plainText = (itineraryText.innerText || itineraryText.textContent).replace(/₹/g, "Rs. ");
        const lines = plainText.split(/\n+/);

        lines.forEach((line) => {
          if (/day\s*\d+/i.test(line)) {
            // Highlight "Day" headers
            doc.setFont("times", "bold");
            doc.setFontSize(14);
            doc.setTextColor(34, 150, 243);
            doc.text(line.trim(), margin, y);
            y += 20;
            doc.setDrawColor(200, 200, 200);
            doc.line(margin, y, 550, y);
            y += 10;
          } else {
            // Regular text
            doc.setFont("times", "normal");
            doc.setFontSize(12);
            doc.setTextColor(0, 0, 0);

            const wrapped = doc.splitTextToSize(line.trim(), 500);
            wrapped.forEach((subLine) => {
              if (y > pageHeight - 60) {
                doc.addPage();
                y = margin;
              }
              doc.text(subLine, margin, y);
              y += 16;
            });
          }
        });

        // --- FOOTER ---
        doc.setFontSize(10);
        doc.setTextColor(120);
        doc.text(
          "Generated using Voyager AI",
          margin,
          pageHeight - 20
        );

        // Save file
        doc.save(`Travel_Itinerary_${destination}.pdf`);
      };
    } catch (err) {
      console.error("Error:", err);
      itineraryText.innerHTML = `<p style="color:red;">⚠️ ${err.message}</p>`;
      downloadBtn.classList.add("hidden");
    }
  });

  regenerateBtn.addEventListener("click", async () => {
    if (!lastFormData) return;

    const selectedChipEls = Array.from(feedbackChips).filter((c) =>
      c.classList.contains("selected")
    );
    const selectedChipKeys = selectedChipEls.map((c) => c.dataset.value);
    const selectedChipLabels = selectedChipEls.map((c) => c.textContent);
    const freeText = feedbackText.value.trim();

    const feedbackNote = [...selectedChipLabels, freeText].filter(Boolean).join("; ");

    itineraryText.innerHTML = `<p>Regenerating with your feedback…</p>`;
    downloadBtn.classList.add("hidden");

    try {
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastFormData, feedback: feedbackNote, feedbackChips: selectedChipKeys }),
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();

      let text = data.itinerary || "No itinerary generated.";
      text = text
        .replace(/\*\*/g, "")
        .replace(/#+/g, "<br><br><strong>")
        .replace(/\n{2,}/g, "<br><br>")
        .replace(/\n/g, "<br>")
        .replace(/- /g, "• ")
        .replace(/\|/g, " | ")
        .replace(/---/g, "<hr>")
        .replace(/> /g, "")
        .replace(/\*/g, "")
        .trim();

      itineraryText.innerHTML = `<div class="itinerary-box">${text}</div>`;
      downloadBtn.classList.remove("hidden");

      lastBudgetDeltas = data.budgetAdjustment ? data.budgetAdjustment.deltas : null;
      budgetBreakdownDiv.innerHTML = renderBudgetBreakdown(Number(lastFormData.budget), lastBudgetDeltas);
      if (data.budgetAdjustment && data.budgetAdjustment.label) {
        budgetBreakdownDiv.innerHTML =
          `<p class="weather-note">Budget adjusted for: <strong>${data.budgetAdjustment.label}</strong></p>` +
          budgetBreakdownDiv.innerHTML;
      }

      feedbackChips.forEach((c) => c.classList.remove("selected"));
      feedbackText.value = "";
    } catch (err) {
      console.error("Error:", err);
      itineraryText.innerHTML = `<p style="color:red;">⚠️ ${err.message}</p>`;
    }
  });
});
