# Chromatic Engine — Genetic Algorithm Palette Optimizer

Chromatic Engine is a high-precision palette optimization engine powered by a custom Genetic Algorithm (GA).  
It uses perceptual color models (LAB, XYZ, HSV), CIEDE2000 color difference metrics, and mode-aware transformations to generate professional-grade color palettes suitable for branding, UI design, visualization themes, and creative workflows.

The frontend is built with HTML, CSS, and JavaScript, featuring an animated chromosome banner, typing-effect title, polished UI, and smooth palette and chart components.  
The backend is implemented using Python and Flask, serving palette generation and GA convergence data.

---

## Key Features

### Genetic Algorithm Engine
- Multi-objective fitness scoring using CIEDE2000
- Diversity preservation and elite selection
- Mode-aware mutation and crossover
- Base-color anchoring (optional)
- Full convergence history for analysis

### Color Science Core
- RGB ↔ XYZ ↔ LAB conversions
- CIEDE2000 perceptual distance
- Mode transformations:  
  Pastel, Vibrant, Neon, Earthy, Monochrome, Gradient, Balanced

### Frontend UI
- Minimal, modern, gradient-based interface
- Animated chromosome banner
- Typing-effect project title
- Animated color swatches with hover lift
- Professional hex display
- Convergence chart (Chart.js)

### Backend API
- Lightweight Flask API
- Predictable JSON output
- Supports multiple palette sizes and GA settings

---

## Project Structure

---

## Installation & Running

### 1. Backend (Flask API)

Navigate to backend:

cd backend
pip install -r requirements.txt
python app.py


The backend starts on:

http://127.0.0.1:5000

### 2. Frontend (Static Server)

Navigate to frontend:

cd frontend
python -m http.server 8000


Open your browser:

http://localhost:8000

### API Endpoints
## Generate Optimized Palette
POST /generate

## Body:

{
  "n_colors": 5,
  "mode": "vibrant",
  "base_color": "#ff8800",
  "pop_size": 120,
  "generations": 60
}


## Response:

{
  "palettes": [
    {
      "hex": ["#FFE7B0", "#FF8850", "#F63939", "#A51C1C", "#4B0C0C"],
      "score": 1.148
    }
  ]
}

### Get Convergence Data
POST /generate_history


Returns:

max fitness per generation

mean fitness

min fitness

best palette snapshot

Used to plot the fitness chart.

### Technologies Used
## Frontend

HTML5, CSS3, JavaScript

Chart.js

Responsive design

Typing animation and CSS keyframe animations

## Backend

Python 3

Flask

NumPy

Custom LAB/CIEDE2000 implementation

### Recommended Use Cases

Brand palette creation

UI theme generation

Data visualization color systems

Web/graphic design tools

### Acknowledgements

This project was built as a complete, modular demonstration of applying Genetic Algorithms to color optimization, showcasing both computational design and frontend engineering.

Educational demos for GAs and color theory

