from flask import Flask, jsonify, send_from_directory
import pandas as pd
import threading
import time
import torch
import torch.nn as nn
import os
import re
from threading import Lock
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# =========================
# FLASK APP
# =========================
app = Flask(__name__, static_folder="../frontend", static_url_path="")

# =========================
# CONFIG
# =========================
YOUTUBE_API_KEY = "AIzaSyBhRW_3Nvx_p_9Evkzo0wK3vrCYihMAfAI"
MAX_COMMENTS_PER_VIDEO = 50

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
VIDEO_CSV_PATH = os.path.join(DATASET_DIR, "video_ids.csv")

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_FILE = os.path.join(MODEL_DIR, "xlm_roberta_bilstm_mha.pt")

# =========================
# LOAD DATA
# =========================
video_df = pd.read_csv(VIDEO_CSV_PATH)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# =========================
# MODEL (MATCH TRAINING EXACTLY)
# =========================
class XLMRobertaBiLSTM_MHA(nn.Module):

    def __init__(self, num_labels=2):
        super().__init__()

        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, input_ids=None, attention_mask=None):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state

        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)

        # IMPORTANT: SAME AS TRAINING
        context = torch.mean(attn_out, dim=1)

        logits = self.classifier(self.dropout(context))

        return logits


# =========================
# LOAD MODEL
# =========================
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)

model = XLMRobertaBiLSTM_MHA()

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model not found at {MODEL_FILE}")

# IMPORTANT: exact loading (NO strict=False)
model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# GLOBAL RESULTS
# =========================
lock = Lock()

RESULTS = {
    "status": "processing",
    "total_videos": len(video_df),
    "processed_videos": 0,
    "total_comments": 0,
    "abusive_count": 0,
    "all_comments": []
}

# =========================
# PREPROCESS
# =========================
def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# =========================
# MODEL PREDICTION
# =========================
def model_predict(text):

    text = preprocess_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs, dim=1)
    toxic_prob = probs[0][1].item()

    # REAL classification (same rule)
    is_toxic = toxic_prob >= 0.9

    return is_toxic, round(toxic_prob, 4)

# =========================
# FETCH COMMENTS
# =========================
def fetch_comments(video_id):

    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=MAX_COMMENTS_PER_VIDEO,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "text": s["textDisplay"],
                "author": s.get("authorDisplayName", "Unknown"),
                "profile_url": s.get("authorChannelUrl", "#")
            })

    except HttpError:
        pass

    return comments

# =========================
# PROCESS
# =========================
def process_all_videos():

    for vid in video_df["video_id"].dropna():

        comments = fetch_comments(vid)

        with lock:
            RESULTS["total_comments"] += len(comments)

        for c in comments:

            pred, prob = model_predict(c["text"])

            with lock:
                RESULTS["all_comments"].append({
                    "video_id": vid,
                    "comment_text": c["text"],
                    "author": c["author"],
                    "profile_url": c["profile_url"],
                    "score": prob,
                    "is_abusive": pred
                })

                if pred:
                    RESULTS["abusive_count"] += 1

        with lock:
            RESULTS["processed_videos"] += 1

        time.sleep(0.2)

    RESULTS["status"] = "completed"

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/abusive-comments")
def abusive_comments():
    return jsonify(RESULTS)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    threading.Thread(target=process_all_videos, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)