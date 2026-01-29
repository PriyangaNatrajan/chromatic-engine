/* frontend/app.js
   Matches backend endpoints defined in backend/app.py
   Uses RELATIVE URLs for PythonAnywhere deployment
*/

const TITLE = "Chromatic Engine";
const typed = document.getElementById("typed-title");

/* Typing effect */
(function typeTitle(){
  let i = 0, speed = 45;
  function step(){
    typed.textContent = TITLE.slice(0, i);
    i++;
    if (i <= TITLE.length) setTimeout(step, speed);
  }
  step();
})();

/* DOM references */
const generateBtn = document.getElementById("generateBtn");
const historyBtn = document.getElementById("historyBtn");
const palettesContainer = document.getElementById("palettes");
const baseColorInput = document.getElementById("baseColor");
const paletteSizeInput = document.getElementById("paletteSize");
const popInput = document.getElementById("popSize");
const gensInput = document.getElementById("generations");
const modeSelect = document.getElementById("modeSelect");

const chartSection = document.getElementById("chartSection");
const fitnessCanvas = document.getElementById("fitnessChart");
const closeChart = document.getElementById("closeChart");
let fitnessChart = null;

/* Render palettes */
function renderPalettes(list){
  palettesContainer.innerHTML = "";
  list.forEach(p => {
    const wrapper = document.createElement("div");
    wrapper.className = "palette";

    const swatches = document.createElement("div");
    swatches.className = "swatches";

    p.hex.forEach(h => {
      const s = document.createElement("div");
      s.className = "swatch";
      s.style.background = h;
      s.title = h;
      s.addEventListener("click", () => {
        navigator.clipboard?.writeText(h);
        s.style.outline = "3px solid rgba(255,255,255,0.14)";
        setTimeout(() => s.style.outline = "", 800);
      });
      swatches.appendChild(s);
    });

    const hexes = document.createElement("div");
    hexes.className = "hexes";
    p.hex.forEach(h => {
      const el = document.createElement("div");
      el.className = "hex";
      el.textContent = h;
      hexes.appendChild(el);
    });

    wrapper.appendChild(swatches);
    wrapper.appendChild(hexes);
    palettesContainer.appendChild(wrapper);
  });
}

/* Call backend /generate */
async function fetchGenerate(baseColor = null){
  const payload = {
    mode: modeSelect.value || "default",
    base_color: baseColor,
    n_colors: parseInt(paletteSizeInput.value, 10),
    pop_size: parseInt(popInput.value, 10),
    generations: parseInt(gensInput.value, 10)
  };

  try{
    const res = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const t = await res.text();
      console.error("Server error:", t);
      alert("Server error while generating palettes.");
      return;
    }

    const data = await res.json();
    if (!data.palettes) {
      console.error("Unexpected response:", data);
      alert("Unexpected server response.");
      return;
    }

    renderPalettes(data.palettes);

  } catch (err) {
    console.error("Network error:", err);
    alert("Network error while contacting the server.");
  }
}

/* Call backend /generate_history */
async function fetchHistory(baseColor = null){
  const payload = {
    mode: modeSelect.value || "default",
    base_color: baseColor,
    n_colors: parseInt(paletteSizeInput.value, 10),
    pop_size: parseInt(popInput.value, 10),
    generations: parseInt(gensInput.value, 10)
  };

  try{
    const res = await fetch("/generate_history", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const t = await res.text();
      console.error("History error:", t);
      alert("Server error while fetching history.");
      return;
    }

    const json = await res.json();
    if (!Array.isArray(json.history)) {
      console.error("Invalid history:", json);
      return;
    }

    showChart(json.history);

  } catch (err) {
    console.error("History fetch failed:", err);
    alert("Network error retrieving history.");
  }
}

/* Chart rendering */
function showChart(history){
  chartSection.classList.remove("hidden");

  const labels = history.map(h => `Gen ${h.generation}`);
  const maxs = history.map(h => h.max_fitness);
  const means = history.map(h => h.mean_fitness);
  const mins = history.map(h => h.min_fitness);

  const ctx = fitnessCanvas.getContext("2d");
  if (fitnessChart) fitnessChart.destroy();

  fitnessChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Max Fitness", data: maxs, borderColor: "#3be6d1", tension: 0.25, fill: false },
        { label: "Mean Fitness", data: means, borderColor: "#7aa8ff", tension: 0.25, fill: false },
        { label: "Min Fitness", data: mins, borderColor: "#ffa76b", tension: 0.25, fill: false }
      ]
    }
  });
}

/* Event bindings */
generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  await fetchGenerate(baseColorInput.value || null);
  generateBtn.disabled = false;
  generateBtn.textContent = "Generate Palette";
});

historyBtn.addEventListener("click", async () => {
  historyBtn.disabled = true;
  historyBtn.textContent = "Loading...";
  await fetchHistory(baseColorInput.value || null);
  historyBtn.disabled = false;
  historyBtn.textContent = "Show Convergence";
});

closeChart.addEventListener("click", () => {
  chartSection.classList.add("hidden");
  if (fitnessChart) fitnessChart.destroy();
});

/* Initial placeholder */
document.addEventListener("DOMContentLoaded", () => {
  renderPalettes([{
    hex: ["#E1B0E8", "#FF00FF", "#000000", "#FFFF00", "#00FFFF"],
    score: 0.0
  }]);
});
