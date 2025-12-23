import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')  # Good for mouth detection

class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Face box (blue)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            
            # Eyes (green)
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Mouth/smile (red) - focus on lower half for better accuracy
            mouth_region = roi_gray[int(h/2):h, :]
            mouths = smile_cascade.detectMultiScale(mouth_region, scaleFactor=1.7, minNeighbors=20)
            for (mx, my, mw, mh) in mouths:
                my += int(h/2)
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
            
            # Emotion detection (on face crop)
            try:
                analysis = DeepFace.analyze(img[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False, silent=True)
                emotion = analysis[0]['dominant_emotion'].capitalize()
                cv2.putText(img, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            except Exception as e:
                cv2.putText(img, "No face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return img

st.title("🔥 Live Emotion Detection from Eyes & Mouth 🔥")
st.write("Open your webcam → make faces → see your emotion + highlighted eyes/mouth!")

RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19305"]}]})

webrtc_streamer(
    key="emotion-key",
    video_transformer_factory=EmotionTransformer,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19305"]}]}),
)
