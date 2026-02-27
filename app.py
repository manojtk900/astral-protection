
# app.py (FULL - ONLY MODEL LOADING FIXED)

import os
import re
import traceback
print("CURRENT WORKING DIRECTORY:", os.getcwd())

from urllib.parse import urlparse, unquote
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, make_response
)

try:
    import pickle
    import sklearn   # 🔥 Required to load trained model
except Exception:
    pickle = None

# 🔥 Base directory fix (important for loading models correctly)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# -----------------------
# Absolute model paths
# -----------------------
CANDIDATE_VECTOR_NAMES = [
    os.path.join(BASE_DIR, "models", "vectorizer_lr_new.pkl"),
]

CANDIDATE_MODEL_NAMES = [
    os.path.join(BASE_DIR, "models", "model_lr_new.pkl"),
]

vector = None
model = None
VECTOR_PATH = None
MODEL_PATH = None

# -----------------------
# Pickle loader
# -----------------------
def try_load_pickle(path):
    if not os.path.exists(path):
        return None, f"not_found:{path}"
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"load_error:{path}:{e}\n{tb}"

def find_and_load():
    global vector, model, VECTOR_PATH, MODEL_PATH

    if pickle is None:
        print("❌ Pickle not available")
        return

    # Load vectorizer
    for p in CANDIDATE_VECTOR_NAMES:
        obj, err = try_load_pickle(p)
        if obj is not None:
            vector = obj
            VECTOR_PATH = p
            print("✅ Vectorizer loaded from:", p)
            break
        else:
            print("Vectorizer issue:", err)

    # Load model
    for p in CANDIDATE_MODEL_NAMES:
        obj, err = try_load_pickle(p)
        if obj is not None:
            model = obj
            MODEL_PATH = p
            print("✅ Model loaded from:", p)
            break
        else:
            print("Model issue:", err)

find_and_load()

# -----------------------
# URL normalization
# -----------------------
def normalize_url_for_model(url: str):
    if not url or not isinstance(url, str):
        return ("", "")
    u = url.strip()
    try:
        u = unquote(u)
    except Exception:
        pass
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u
    parsed = urlparse(u)
    domain = parsed.netloc.split(":", 1)[0].lower().strip()
    path = (parsed.path or "").strip("/")
    if path:
        return domain, f"{domain}/{path}"
    return domain, domain

# -----------------------
# Prediction
# -----------------------
def predict_url(raw_input: str):
    if vector is None or model is None:
        return {"status": "error",
                "message": "Model or vectorizer not loaded. See server logs.",
                "css_class": "neutral"}

    domain_only, domain_plus_path = normalize_url_for_model(raw_input)
    candidates = [domain_plus_path]

    try:
        X = vector.transform(candidates)
        preds = model.predict(X)
        result = str(preds[0]).lower()

        if result in ("0", "safe", "good"):
            return {"status": "safe",
                    "message": "This website looks SAFE ✅",
                    "css_class": "good"}
        else:
            return {"status": "malicious",
                    "message": "Warning — this looks like a PHISHING website! ⚠️",
                    "css_class": "bad"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error",
                "message": "Prediction failed.",
                "css_class": "neutral"}

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        url = (request.form.get("url") or "").strip()

        if not url:
            return redirect(url_for("index"))

        result = predict_url(url)

        return render_template(
            "index.html",
            predict=result,
            submitted_url=url
        )

    return render_template("index.html", predict=None)

@app.route("/interface")
def interface():
    return render_template("interface.html")



@app.route("/health")
def health():
    import os
    return jsonify({
        "ok": True,
        "cwd": os.getcwd(),
        "models_folder_exists": os.path.exists("models"),
        "vector_file_exists": os.path.exists("models/vectorizer_lr_new.pkl"),
        "model_file_exists": os.path.exists("models/model_lr_new.pkl"),
        "vector_loaded": bool(vector),
        "model_loaded": bool(model)
    })

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/home")
def home():
    return render_template("home.html")


#--------------
@app.route("/admin/model-info")
def model_info():
    return jsonify({
        "vector_loaded": bool(vector),
        "model_loaded": bool(model),
        "meta": {
            "model_type": "Logistic Regression"
        }
    })

#--------

@app.route("/chat_api", methods=["POST"])
def chat_api():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip().lower()

    if not message:
        return jsonify({"reply": "Please type a message."})

    # -------- Commands --------

    if message == "model info":
        return jsonify({
            "reply": f"""
Model Loaded: {bool(model)}
Vector Loaded: {bool(vector)}
Model Type: Logistic Regression
"""
        })

    if message == "help":
        return jsonify({
            "reply": """
Available Commands:
- model info
- help
- phishing tips
- email signs
Or paste a URL.
"""
        })

    if "phishing" in message:
        return jsonify({
            "reply": """
Phishing Red Flags:
• Suspicious domain names
• Too many hyphens
• Urgent login pages
• Fake bank emails
"""
        })

    if "email" in message:
        return jsonify({
            "reply": """
Email Phishing Signs:
• Unknown sender
• Urgent request
• Fake attachments
• Suspicious links
"""
        })

    # -------- URL Detection --------
    if message.startswith("http") or "." in message:
        result = predict_url(message)
        return jsonify({"reply": result.get("message")})

    return jsonify({"reply": "Hello! Paste a URL to check if it's phishing."})


# Run
# -----------------------
@app.route("/python")
def python_version():
    import sys
    return {"python": sys.version}

# -----------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))