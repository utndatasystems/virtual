from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

LEADERBOARD_FILE = "leaderboard.json"


def read_leaderboard():
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def write_leaderboard(data):
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route("/leaderboard", methods=["GET"])
def get_leaderboard():
    return jsonify(read_leaderboard())


@app.route("/leaderboard", methods=["POST"])
def post_leaderboard():
    data = request.get_json(force=True, silent=True) or {}
    data["submitTime"] = data.get(
        "submitTime",
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    leaderboard = read_leaderboard()
    leaderboard.append(data)
    write_leaderboard(leaderboard)

    return jsonify({"status": "ok"}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)