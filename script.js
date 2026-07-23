document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("trip-form");
  const outputDiv = document.getElementById("output");
  const itineraryText = document.getElementById("itinerary-text");
  const downloadBtn = document.getElementById("download-pdf");
  const budgetBreakdownDiv = document.getElementById("budget-breakdown");

  function getBudgetBreakdown(totalBudget) {
    const categories = [
      { label: "Accommodation", percent: 0.35 },
      { label: "Food & Dining", percent: 0.25 },
      { label: "Local Transport", percent: 0.15 },
      { label: "Activities & Sightseeing", percent: 0.15 },
      { label: "Shopping & Misc.", percent: 0.10 },
    ];
    return categories.map((c) => ({
      label: c.label,
      amount: totalBudget * c.percent,
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

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const destination = document.getElementById("destination").value.trim();
    const budget = document.getElementById("budget").value.trim();
    const days = document.getElementById("days").value.trim();
    const preferences = document.getElementById("preferences").value.trim();

    // Show loading
    outputDiv.classList.remove("hidden");
    itineraryText.innerHTML = `<p>Generating your itinerary…</p>`;
    downloadBtn.classList.add("hidden");

    if (!destination || !budget || !days) {
      itineraryText.innerHTML = `<p style="color:red;">Please fill in all required fields.</p>`;
      return;
    }

    try {
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ destination, budget, days, preferences }),
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const data = await response.json();

      budgetBreakdownDiv.innerHTML = renderBudgetBreakdown(Number(budget));

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

        const pageHeight = 842;
        const margin = 40;
        let y = 160;

        // --- BUDGET BREAKDOWN ---
        doc.setFont("times", "bold");
        doc.setFontSize(13);
        doc.setTextColor(30, 77, 91);
        doc.text("Budget Breakdown", margin, y);
        y += 18;

        const breakdown = getBudgetBreakdown(Number(budget));
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
});
