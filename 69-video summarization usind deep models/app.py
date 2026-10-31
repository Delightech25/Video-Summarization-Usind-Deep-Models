import os
import cv2
import shutil
import torch
import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for, flash
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from PIL import Image

# -----------------------------
# APP CONFIG
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
FRAMES_DIR = "static/frames"
USERS_FILE = "users.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create users.csv if not exists
if not os.path.exists(USERS_FILE):
    pd.DataFrame({"username": [], "password": []}).to_csv(USERS_FILE, index=False)

# -----------------------------
# LOAD MODELS ONCE (IMPORTANT)
# -----------------------------
print("Loading models... Please wait")

# Caption model
caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name)
caption_processor = ViTImageProcessor.from_pretrained(caption_model_name)
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)

caption_model.to("cpu")
caption_model.eval()

# Summarization model
summary_model_name = "sshleifer/distilbart-cnn-12-6"
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

summary_model.to("cpu")
summary_model.eval()

print("Models loaded successfully.")

# -----------------------------
# FRAME EXTRACTION
# -----------------------------
def extract_frames(video_path, interval=3):
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        raise Exception("Invalid video file")

    frame_interval = fps * interval
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            cv2.imwrite(f"{FRAMES_DIR}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()
    return saved


# -----------------------------
# CAPTION GENERATION
# -----------------------------
def generate_captions():
    captions = []

    for img_name in sorted(os.listdir(FRAMES_DIR)):
        img_path = os.path.join(FRAMES_DIR, img_name)
        image = Image.open(img_path).convert("RGB")

        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = caption_model.generate(pixel_values, max_length=20)

        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)

    return captions


# -----------------------------
# TEXT SUMMARIZATION
# -----------------------------
def summarize_text(text):
    text = text[:3000]

    inputs = summary_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = summary_model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4
        )

    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv(USERS_FILE)

        if ((users["username"] == username) &
            (users["password"] == password)).any():

            session["user"] = username
            return redirect("/dashboard")

        flash("Invalid credentials")

    return render_template("login.html")


# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv(USERS_FILE)

        if (users["username"] == username).any():
            flash("Username already exists")
            return redirect("/register")

        new_user = pd.DataFrame({
            "username": [username],
            "password": [password]
        })

        users = pd.concat([users, new_user])
        users.to_csv(USERS_FILE, index=False)

        return redirect("/")

    return render_template("register.html")


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    return render_template("dashboard.html", user=session["user"])


# =========================
# VIDEO SUMMARIZATION
# =========================
@app.route("/summarize", methods=["POST"])
def summarize():
    if "user" not in session:
        return redirect("/")

    if "video" not in request.files:
        flash("No video uploaded")
        return redirect("/dashboard")

    file = request.files["video"]

    if file.filename == "":
        flash("No selected file")
        return redirect("/dashboard")

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Process video
    extract_frames(video_path)
    captions = generate_captions()

    combined_text = " ".join(captions)
    summary = summarize_text(combined_text)

    return render_template(
        "result.html",
        original_video=file.filename,
        text_summary=summary
    )


# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True, use_reloader=False)