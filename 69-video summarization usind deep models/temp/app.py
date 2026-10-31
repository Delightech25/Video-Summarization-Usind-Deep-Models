import os
import cv2
import shutil
import torch
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from PIL import Image
from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "secretkey"

UPLOAD_FOLDER = "static/uploads"
FRAMES_DIR = "frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
# CAPTION MODEL
# -----------------------------
def generate_captions():
    model_name = "nlpconnect/vit-gpt2-image-captioning"

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to("cpu")
    model.eval()

    captions = []

    for img_name in sorted(os.listdir(FRAMES_DIR)):
        img_path = os.path.join(FRAMES_DIR, img_name)
        image = Image.open(img_path).convert("RGB")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=20)

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        captions.append(caption)

    return captions


# -----------------------------
# SUMMARIZATION MODEL
# -----------------------------
def summarize_text(text):
    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.to("cpu")
    model.eval()

    text = text[:3000]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# =========================
# LOGIN
# =========================

@app.route("/", methods=["GET","POST"])
def login():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        users=pd.read_csv("users.csv")

        if ((users['username']==username) &
            (users['password']==password)).any():

            session["user"]=username

            return redirect("/dashboard")

    return render_template("login.html")


# =========================
# REGISTER
# =========================

@app.route("/register",methods=["GET","POST"])
def register():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        users=pd.read_csv("users.csv")

        new_user=pd.DataFrame({
            "username":[username],
            "password":[password]
        })

        users=pd.concat([users,new_user])

        users.to_csv("users.csv",index=False)

        return redirect("/")

    return render_template("register.html")


# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    return render_template(
        "dashboard.html",
        user=session["user"]
    )



@app.route("/summarize", methods=["POST"])
def summarize():
    if "video" not in request.files:
        flash("No video uploaded")
        return redirect(url_for("index"))

    file = request.files["video"]

    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))

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


@app.route("/logout")
def logout():

    session.pop("user",None)

    return redirect("/")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True, use_reloader=False)