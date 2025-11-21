# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from .ga import PaletteGA

app = Flask(__name__)
CORS(app)   # Enable CORS for all routes


@app.route("/")
def home():
    return "GA Palette Generator API is running. Use /generate"


@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json(silent=True) or {}

    # Read params from frontend
    mode = payload.get("mode", "default")
    base_color = payload.get("base_color", None)

    # Default GA params
    n_colors = 5
    pop_size = 120
    generations = 80
    elite = 2
    mutation_rate = 0.32

    ga = PaletteGA(
        n_colors=n_colors,
        pop_size=pop_size,
        elite_size=elite,
        mutation_rate=mutation_rate,
        generations=generations,
        seed=None,
        mode=mode,
        base_color=base_color
    )

    palettes = ga.get_top_palettes_hex(top_k=6)
    return jsonify({"palettes": palettes})
@app.route("/generate_history", methods=["POST"])
def generate_history():
    payload = request.get_json(silent=True) or {}

    mode = payload.get("mode", "default")
    base_color = payload.get("base_color", None)

    # allow overriding GA params optionally from frontend
    n_colors = int(payload.get("n_colors", 5))
    pop_size = int(payload.get("pop_size", 120))
    generations = int(payload.get("generations", 60))
    elite = int(payload.get("elite", 6))
    mutation_rate = float(payload.get("mutation_rate", 0.18))

    ga = PaletteGA(
        n_colors=n_colors,
        pop_size=pop_size,
        elite_size=elite,
        mutation_rate=mutation_rate,
        generations=generations,
        seed=None,
        mode=mode,
        base_color=base_color
    )

    # run evolve_with_history (method added above)
    history = ga.evolve_with_history()
    return jsonify({"history": history})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
