import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import gc
from math import hypot
import time

# Hiding Streamlit style
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Authentication function
def authenticate(username, password):
    return (username, password) in [("admin", "password"), ("Kumar", "password")]

def process_frame(frame, tap_data, start_time):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lmList = []
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) >= 21:
                index_tip = lmList[8][1], lmList[8][2]
                thumb_tip = lmList[4][1], lmList[4][2]

                cv2.circle(frame, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

                if index_tip is not None and thumb_tip is not None:
                    distance_pixels = calculate_distance(thumb_tip, index_tip)
                    distance_cm = distance_pixels * 0.1

                    tap_data.append({
                        'Time': time.time() - start_time,
                        'Distance (pixels)': distance_pixels,
                        'Distance (cm)': distance_cm
                    })

    return frame

# Define Streamlit app
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.success("Logged in successfully")
            else:
                st.error("Invalid username or password")
        return

    st.title("Finger Tap Detection")

    st.header("Upload Video for Analysis")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov"])
    start_button = st.button("Start Analysis")

    if video_file is not None and start_button:
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_file.write(video_file.read())
        temp_video_file.close()

        cap = cv2.VideoCapture(temp_video_file.name)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)

        stframe = st.empty()
        tap_data = []
        start_time = time.time()

        frame_skip = 5
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = process_frame(frame, tap_data, start_time)
            stframe.image(frame, channels="BGR")

            frame = None
            gc.collect()

        cap.release()
        os.remove(temp_video_file.name)

        st.write(f"Processed {len(tap_data)} frames.")

if __name__ == "__main__":
    main()
