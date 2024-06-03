import streamlit as st
import cv2
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
from math import hypot
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

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

def main():
    st.title("Finger Tap Detection")

    # User registration
    st.header("Patient Registration")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    submit_button = st.button("Submit")

    if submit_button:
        st.success(f"Registered: {name}, {age}, {sex}")

    # Get user input for video file upload
    st.header("Upload Video for Analysis")
    video_file = st.file_uploader("Upload a video file", type=["mp4"])
    start_button = st.button("Start Analysis")

    if video_file is not None and start_button:
        # Save the uploaded file to disk
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("uploaded_video.mp4")  # Use the file path as input to VideoCapture

        stframe = st.empty()
        graph_placeholder = st.empty()

        # Initialize variables
        start_time = time.time()
        speeds_graph = []
        tap_count = 0
        tap_data = []
        tap_timestamps = []  # List to store timestamps of each tap

        # Thresholds
        initial_touch_threshold = 50  # Adjust sensitivity for initial touch
        separation_threshold = 50  # Adjust sensitivity for separation

        hand_start_position = None
        tap_detected = False

        fig, ax = plt.subplots()  # Create figure and axis objects

        try:  # Introducing try...finally block for resource cleanup
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    st.warning("No frame to read from the video. Exiting.")
                    break

                # Reduce the resolution of the frame
                img = cv2.resize(img, (640, 360))

                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Process hand landmarks
                results = hands.process(imgRGB)

                lmList = []
                index_speed = 0
                thumb_tip = None
                index_tip = None

                if results.multi_hand_landmarks:
                    for handlandmark in results.multi_hand_landmarks:
                        for id, lm in enumerate(handlandmark.landmark):
                            h, w, _ = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([id, cx, cy])

                        if len(lmList) >= 21:
                            index_tip = lmList[8][1], lmList[8][2]
                            thumb_tip = lmList[4][1], lmList[4][2]

                            cv2.circle(img, index_tip, 10, (0, 255, 0), cv2.FILLED)
                            cv2.circle(img, thumb_tip, 10, (0, 255, 0), cv2.FILLED)

                            if index_tip is not None and thumb_tip is not None:
                                # Draw a line between index finger and thumb
                                cv2.line(img, index_tip, thumb_tip, (255, 0, 0), 2)

                                # Calculate the distance between the thumb and index finger
                                distance_pixels = calculate_distance(thumb_tip, index_tip)
                                distance_cm = distance_pixels * 0.1  # Placeholder conversion factor (adjust as needed)

                                # Format the distance value
                                distance_formatted = "{:.2f} cm".format(distance_cm)  # Example format: two decimal places

                                # Tap detection logic
                                if not tap_detected and distance_pixels < initial_touch_threshold:
                                    tap_detected = True
                                    hand_start_position = index_tip
                                    tap_timestamps.append(time.time())  # Record timestamp of the tap

                                if tap_detected and distance_pixels > separation_threshold:
                                    tap_detected = False
                                    tap_count += 1
                                    st.write(f"Tap {tap_count} detected! Distance: {distance_formatted}")

                                    # Save data
                                    tap_data.append({
                                        'Tap Count': tap_count,
                                        'Time': time.time() - start_time,
                                        'Distance (pixels)': distance_pixels,
                                        'Distance (cm)': distance_cm,
                                        'Formatted Distance': distance_formatted,
                                        'Start Position': hand_start_position
                                    })

                                    # Update the plot
                                    speeds_graph.append(distance_pixels)
                                    ax.clear()
                                    ax.plot(speeds_graph, color='b')  # Plot on existing axis
                                    ax.set_title('Finger Tap Distance Over Time')
                                    ax.set_xlabel('Frames')
                                    ax.set_ylabel('Distance (pixels)')
                                    graph_placeholder.pyplot(fig)
                                    plt.close(fig)  # Close the figure to release memory

                stframe.image(img, channels="BGR")
        finally:
            cap.release()  # Release the video capture object
            
            try:
                cv2.destroyAllWindows()  # Close all OpenCV windows
            except cv2.error as e:
                print("Error occurred while closing OpenCV windows:", e)

if __name__ == "__main__":
    main()
