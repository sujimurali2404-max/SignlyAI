import os
import threading
from pathlib import Path

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from gtts import gTTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Sign Language AI", page_icon="🤟", layout="wide")
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Hello", "Yes", "No"]
SEQUENCE_LENGTH = 12
MIN_CONFIDENCE = 0.65
STABLE_SECONDS = 0.55
DATA_FILE = Path("sign_samples.npz")
MODEL_FILE = Path("sign_model.joblib")

# ------------------------------------------------------------
# AI FEATURE EXTRACTION (retained from your original project)
# ------------------------------------------------------------
def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32,
    )
    points -= points[0].copy()
    scale = np.linalg.norm(points[9])
    if scale < 1e-6:
        return None
    points /= scale
    if not np.isfinite(points).all():
        return None
    return points.flatten()


def load_training_data():
    if not DATA_FILE.exists():
        return [], []
    try:
        with np.load(DATA_FILE, allow_pickle=False) as saved:
            X = saved["X"].astype(np.float32)
            y = saved["y"].astype(str)
        if X.ndim != 2 or X.shape[1] != SEQUENCE_LENGTH * 63 or len(X) != len(y):
            return [], []
        return list(X), list(y)
    except Exception:
        return [], []


def save_training_data(X, y):
    np.savez_compressed(
        DATA_FILE,
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=str),
    )


def train_classifier(X, y):
    if len(X) == 0:
        raise ValueError("No training samples have been collected.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=str)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Collect samples for at least two different signs.")
    if np.min(counts) < 2:
        raise ValueError("Collect at least two samples for every sign before training.")

    accuracy = None
    n_classes = len(classes)
    n_test = max(n_classes, int(np.ceil(0.20 * len(y))))
    n_test = min(n_test, len(y) - n_classes)

    if n_test >= n_classes:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=n_test, random_state=42, stratify=y
            )
            validation_model = RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )
            validation_model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, validation_model.predict(X_test))
        except ValueError:
            accuracy = None

    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model, accuracy


# ------------------------------------------------------------
# REAL-TIME BROWSER CAMERA PROCESSOR
# ------------------------------------------------------------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.drawer = mp.solutions.drawing_utils
        self.model = None
        self.selected_label = "A"
        self.collect_requested = False
        self.collect_frames = []
        self.recent_frames = []
        self.X, self.y = load_training_data()
        self.prediction = "Waiting for hand"
        self.confidence = 0.0
        self.latest_spoken = ""
        self.candidate = None
        self.candidate_since = 0.0
        self.stable_prediction = ""
        self.sample_message = ""
        self.mode = "TEST" if MODEL_FILE.exists() else "COLLECT"

        if MODEL_FILE.exists():
            try:
                loaded = joblib.load(MODEL_FILE)
                if getattr(loaded, "n_features_in_", None) == SEQUENCE_LENGTH * 63:
                    self.model = loaded
                    self.mode = "TEST"
            except Exception:
                self.model = None

    def request_sample(self, label):
        with self.lock:
            self.selected_label = label
            self.collect_requested = True
            self.collect_frames = []
            self.sample_message = f"Collecting {label}: hold the sign steadily."

    def set_model(self, model):
        with self.lock:
            self.model = model
            self.mode = "TEST"
            self.recent_frames = []
            self.candidate = None
            self.latest_spoken = ""

    def snapshot(self):
        with self.lock:
            return {
                "prediction": self.prediction,
                "confidence": self.confidence,
                "samples": len(self.X),
                "labels": sorted(set(self.y)),
                "sample_message": self.sample_message,
                "stable_prediction": self.stable_prediction,
                "mode": self.mode,
            }

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        feature = None
        now = __import__("time").monotonic()

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
            feature = extract_features(hand)
        else:
            self.recent_frames.clear()
            self.candidate = None

        if feature is not None:
            self.recent_frames.append(feature)
            self.recent_frames = self.recent_frames[-SEQUENCE_LENGTH:]

        with self.lock:
            if self.collect_requested and feature is not None:
                self.collect_frames.append(feature)
                self.sample_message = (
                    f"Collecting {self.selected_label}: "
                    f"{len(self.collect_frames)}/{SEQUENCE_LENGTH} frames"
                )
                if len(self.collect_frames) >= SEQUENCE_LENGTH:
                    sample = np.asarray(
                        self.collect_frames[:SEQUENCE_LENGTH], dtype=np.float32
                    ).flatten()
                    self.X.append(sample)
                    self.y.append(self.selected_label)
                    save_training_data(self.X, self.y)
                    self.collect_requested = False
                    self.collect_frames = []
                    self.sample_message = f"Saved one training sample for {self.selected_label}."
            model = self.model

        prediction = "Waiting for hand"
        confidence = 0.0
        stable_prediction = ""

        if model is not None and len(self.recent_frames) == SEQUENCE_LENGTH:
            sample = np.asarray(self.recent_frames, dtype=np.float32).flatten().reshape(1, -1)
            try:
                probabilities = model.predict_proba(sample)[0]
                best = int(np.argmax(probabilities))
                predicted = str(model.classes_[best])
                confidence = float(probabilities[best])
                if confidence >= MIN_CONFIDENCE:
                    prediction = predicted
                    if self.candidate != predicted:
                        self.candidate = predicted
                        self.candidate_since = now
                    elif now - self.candidate_since >= STABLE_SECONDS:
                        stable_prediction = predicted
                        self.latest_spoken = predicted
                else:
                    prediction = "Uncertain"
                    self.candidate = None
            except Exception:
                prediction = "Prediction error"

        with self.lock:
            self.prediction = prediction
            self.confidence = confidence
            self.stable_prediction = stable_prediction

        cv2.rectangle(image, (8, 8), (560, 112), (25, 25, 25), -1)
        lines = [
            f"Mode: {self.mode}",
            f"Prediction: {prediction}",
            f"Confidence: {confidence:.1%}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                image, line, (20, 35 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ------------------------------------------------------------
# STREAMLIT USER INTERFACE
# ------------------------------------------------------------
st.title("🤟 Sign Language Detection & Speech")
st.caption("Real-time hand-sign recognition using MediaPipe and a Random Forest classifier.")
# Refresh the dashboard so predictions from the camera worker appear without a button click.

if "processor" not in st.session_state:
    st.session_state.processor = SignProcessor()

processor = st.session_state.processor

with st.sidebar:
    st.header("Training controls")
    selected_label = st.selectbox("Choose the sign to collect", LABELS)
    if st.button("📸 Collect one sample", use_container_width=True):
        processor.request_sample(selected_label)
        st.success(f"Hold the sign for {SEQUENCE_LENGTH} frames.")
    st.caption("Collect several samples for each sign, using different hand positions and lighting.")
    if st.button("🧠 Train AI model", use_container_width=True):
        try:
            with st.spinner("Training the Random Forest model..."):
                model, accuracy = train_classifier(processor.X, processor.y)
                processor.set_model(model)
            st.success("Model trained and saved.")
            if accuracy is not None:
                st.metric("Validation accuracy", f"{accuracy:.1%}")
            else:
                st.info("Validation accuracy was not calculated because there were not enough samples.")
        except Exception as exc:
            st.error(str(exc))

    if st.button("🔄 Reload saved model", use_container_width=True):
        if MODEL_FILE.exists():
            try:
                processor.set_model(joblib.load(MODEL_FILE))
                st.success("Saved model loaded.")
            except Exception as exc:
                st.error(f"Could not load model: {exc}")
        else:
            st.warning("No saved model found. Collect samples and train first.")

left, right = st.columns([2, 1])
with left:
    st.subheader("Live camera")
  ctx = webrtc_streamer(
    key="sign-language-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: processor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    },
    async_processing=True,
)
    st.caption("Allow camera access in your browser. Use HTTPS (Streamlit Cloud provides it).")

with right:
    st.subheader("Detection result")
    status = processor.snapshot()
    st.metric("Detected sign", status["prediction"])
    st.metric("Confidence", f'{status["confidence"]:.1%}')
    st.metric("Training samples", status["samples"])
    st.write("**Mode:**", status["mode"])
    if status["sample_message"]:
        st.info(status["sample_message"])
    if status["labels"]:
        st.write("**Signs collected:**", ", ".join(status["labels"]))
    else:
        st.write("No training signs collected yet.")

st.divider()
st.subheader("🔊 Speech output")
status = processor.snapshot()
phrase = status["stable_prediction"]
if phrase and phrase != st.session_state.get("last_audio_phrase", ""):
    try:
        with st.spinner(f"Preparing speech for: {phrase}"):
            from io import BytesIO
            audio_buffer = BytesIO()
            gTTS(text=phrase, lang="en").write_to_fp(audio_buffer)
            audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3", autoplay=True)
        st.session_state.last_audio_phrase = phrase
        st.success(f"Recognized sign: {phrase}")
    except Exception as exc:
        st.warning(f"Speech generation unavailable: {exc}")
elif phrase:
    st.write(f"Recognized: **{phrase}**")
else:
    st.session_state.last_audio_phrase = ""
    st.write("A stable sign prediction will appear here and be converted to speech.")

st.info(
    "Note: This is a prototype. Sign-language recognition depends on the signs used "
    "in your training data. It does not automatically understand every sign-language "
    "sentence or grammar."
)
