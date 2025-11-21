/* frontend app.js
   Matches backend endpoints defined in backend/app.py (POST /generate and POST /generate_history).
   If your backend is on a different host/port change BACKEND_URL below.
*/

const BACKEND_URL = "http://127.0.0.1:5000"; // matches backend/app.py default. :contentReference[oaicite:4]{index=4}

const TITLE = "Chromatic Engine";
const typed = document.getElementById("typed-title");
(function typeTitle(){
  let i=0, speed=45;
  function step(){
    typed.textContent = TITLE.slice(0, i);
    i++;
    if(i <= TITLE.length) setTimeout(step, speed);
  }
  step();
})();

/* DOM refs */
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

/* render placeholder */
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
        setTimeout(()=> s.style.outline = "", 800);
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

/* call backend /generate (POST) - matches backend/app.py payload keys (mode, base_color, optionally n_colors etc.) */
async function fetchGenerate(baseColor=null){
  const payload = {
    mode: modeSelect.value || "default",
    base_color: baseColor || null,
    n_colors: parseInt(paletteSizeInput.value, 10),
    pop_size: parseInt(popInput.value, 10),
    generations: parseInt(gensInput.value, 10)
  };

  try{
    const res = await fetch(`${BACKEND_URL}/generate`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if(!res.ok){
      const t = await res.text();
      console.error("Server error:", t);
      alert("Server error while generating. See console.");
      return;
    }
    const data = await res.json();
    // expect { palettes: [ {hex:[...], score:...}, ... ] }
    if(!data.palettes) { console.error("Unexpected response:", data); alert("Unexpected response."); return; }
    renderPalettes(data.palettes);
  }catch(err){
    console.error("Network error:", err);
    alert("Network error. Is the backend running on port 5000?");
  }
}

/* fetch history from backend /generate_history (POST) */
async function fetchHistory(baseColor=null){
  const payload = {
    mode: modeSelect.value || "default",
    base_color: baseColor || null,
    n_colors: parseInt(paletteSizeInput.value, 10),
    pop_size: parseInt(popInput.value, 10),
    generations: parseInt(gensInput.value, 10)
  };

  try{
    const res = await fetch(`${BACKEND_URL}/generate_history`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    if(!res.ok){
      const t = await res.text();
      console.error("Server error (history):", t);
      alert("Server error while fetching history. See console.");
      return;
    }
    const json = await res.json();
    const history = json.history;
    if(!Array.isArray(history)){ console.error("Invalid history:", json); return; }
    showChart(history);
  }catch(err){
    console.error("History fetch failed:", err);
    alert("Network error retrieving history.");
  }
}

/* Chart rendering */
function showChart(history){
  chartSection.classList.remove("hidden");
  chartSection.setAttribute("aria-hidden","false");

  const labels = history.map(h => `Gen ${h.generation}`);
  const maxs = history.map(h => h.max_fitness ?? h.max ?? null);
  const means = history.map(h => h.mean_fitness ?? h.mean ?? null);
  const mins = history.map(h => h.min_fitness ?? h.min ?? null);

  const ctx = fitnessCanvas.getContext("2d");
  if(fitnessChart) fitnessChart.destroy();

  fitnessChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Max fitness", data: maxs, borderColor: "#3be6d1", tension:0.25, fill:false, pointRadius:0 },
        { label: "Mean fitness", data: means, borderColor: "#7aa8ff", tension:0.25, fill:false, pointRadius:0 },
        { label: "Min fitness", data: mins, borderColor: "#ffa76b", tension:0.25, fill:false, pointRadius:0 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "top", labels:{color:"#cfe9ee"} } },
      scales: { x: { ticks:{color:"#cfe9ee"} }, y: { ticks:{color:"#cfe9ee"} } }
    }
  });
}

/* Event wiring */
generateBtn.addEventListener("click", async () => {
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating...";
  const base = baseColorInput.value || null;
  await fetchGenerate(base);
  generateBtn.disabled = false;
  generateBtn.textContent = "Generate Palette";
});

historyBtn.addEventListener("click", async () => {
  historyBtn.disabled = true;
  historyBtn.textContent = "Loading...";
  const base = baseColorInput.value || null;
  await fetchHistory(base);
  historyBtn.disabled = false;
  historyBtn.textContent = "Show Convergence";
});

closeChart.addEventListener("click", () => {
  chartSection.classList.add("hidden");
  chartSection.setAttribute("aria-hidden","true");
  if(fitnessChart) fitnessChart.destroy();
});

/* initial placeholder palette */
document.addEventListener("DOMContentLoaded", () => {
  renderPalettes([{
    hex:["#E1B0E8","#FF00FF","#000000","#FFFF00","#00FFFF"],
    score: 0.0
  }]);
});
document.addEventListener("DOMContentLoaded", () => {
    const title = "GENETIC PALETTE OPTIMIZER";
    const el = document.getElementById("typing-title");

    let idx = 0;

    function typeWriter() {
        if (idx < title.length) {
            el.textContent += title.charAt(idx);
            idx++;
            setTimeout(typeWriter, 80); // typing speed
        }
    }

    typeWriter();
});
