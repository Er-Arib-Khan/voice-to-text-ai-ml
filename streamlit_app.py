import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import speech_recognition as sr
import numpy as np
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import os
import av

st.set_page_config(page_title="🎙️ Voice to Text Live", layout="centered")
st.title("🎤 Speak Your Complaint (Live from Browser Mic)")

recognizer = sr.Recognizer()
complaints = []

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = b""
        self.result = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.int16).tobytes()
        self.buffer += pcm

        try:
            audio_data = sr.AudioData(self.buffer, 16000, 2)
            text = recognizer.recognize_google(audio_data)
            self.result = text
            self.buffer = b""  # clear buffer once processed
        except (sr.UnknownValueError, sr.RequestError):
            self.result = None
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode="sendonly",
    in_audio_enabled=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.state.playing:
    if webrtc_ctx.audio_processor:
        result = webrtc_ctx.audio_processor.result
        if result:
            st.success(f"📝 Transcribed: {result}")

            # Sentiment analysis
            blob = TextBlob(result)
            polarity = blob.polarity
            subjectivity = blob.subjectivity

            # Save to CSV
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([[now, result, polarity, subjectivity]],
                              columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])
            file_exists = os.path.exists("complaints.csv")
            df.to_csv("complaints.csv", mode='a', header=not file_exists, index=False)

            st.write("**Polarity:**", polarity)
            st.write("**Subjectivity:**", subjectivity)
