import os
import cv2
import shutil
from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = "video.mp4"
FRAMES_DIR = "frames"
FRAME_INTERVAL_SECONDS = 3


# -----------------------------
# STEP 1: Extract Frames
# -----------------------------
def extract_frames(video_path, output_folder, interval=3):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        raise Exception("Could not read video FPS. Check video file.")

    frame_interval = fps * interval
    count = 0
    saved = 0

    print("Extracting frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Saved {saved} frames.")


# -----------------------------
# STEP 2: Caption Frames
# -----------------------------
def generate_captions(frames_folder):
    print("Loading caption model (CPU manual mode)...")

    model_name = "nlpconnect/vit-gpt2-image-captioning"

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to("cpu")
    model.eval()

    captions = []

    for img_name in sorted(os.listdir(frames_folder)):
        img_path = os.path.join(frames_folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values

            with torch.no_grad():
                output_ids = model.generate(
                    pixel_values,
                    max_length=20,
                    num_beams=4
                )

            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            captions.append(caption)
            print(f"{img_name}: {caption}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    return captions


# -----------------------------
# STEP 3: Summarize Text
# -----------------------------
def summarize_text(text):
    print("Loading summarization model (CPU)...")

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

    # Limit input size to avoid memory crash
    text = text[:3000]

    print("Generating summary...")

    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )

    return summary[0]["summary_text"]


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video file not found. Place video.mp4 in same folder.")
        return

    extract_frames(VIDEO_PATH, FRAMES_DIR, FRAME_INTERVAL_SECONDS)

    captions = generate_captions(FRAMES_DIR)

    if not captions:
        print("No captions generated.")
        return

    combined_text = " ".join(captions)

    print("\nCombined Text:\n")
    print(combined_text)

    final_summary = summarize_text(combined_text)

    print("\n==============================")
    print("FINAL SUMMARY:")
    print("==============================\n")
    print(final_summary)


if __name__ == "__main__":
    main()