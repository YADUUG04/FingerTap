import cv2
import streamlit as st
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt
from math import hypot

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

# Define Streamlit app
def main():
    st.title("Finger Tap Detection")

    # Get user input for video file upload
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
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
        tap_speeds = []  # List to store speeds of each tap

        # Thresholds
        initial_touch_threshold = 50  # Adjust sensitivity for initial touch
        separation_threshold = 50  # Adjust sensitivity for separation

        hand_start_position = None
        tap_detected = False

        fig, ax = plt.subplots()  # Create figure and axis objects

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                st.warning("No frame to read from the video. Exiting.")
                break

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

                                # Calculate tap duration
                                tap_end_time = time.time()
                                tap_duration = tap_end_time - tap_timestamps[-1]

                                # Calculate tap speed
                                tap_speed = distance_cm / tap_duration  # cm/second
                                tap_speeds.append(tap_speed)

                                # Save data
                                tap_data.append({
                                    'Tap Count': tap_count,
                                    'Time': tap_end_time - start_time,
                                    'Distance (pixels)': distance_pixels,
                                    'Distance (cm)': distance_cm,
                                    'Formatted Distance': distance_formatted,
                                    'Start Position': hand_start_position,
                                    'Tap Duration': tap_duration,
                                    'Tap Speed (cm/s)': tap_speed
                                })

                                # Update the plot
                                speeds_graph.append(distance_pixels)
                                ax.clear()
                                ax.plot(speeds_graph, color='b')  # Plot on existing axis
                                ax.set_title('Finger Tap Distance Over Time')
                                ax.set_xlabel('Frames')
                                ax.set_ylabel('Distance (pixels)')
                                graph_placeholder.pyplot(fig)

            stframe.image(img, channels="BGR")

        cap.release()

        # Calculate the average tap speed
        average_speed = sum(tap_speeds) / len(tap_speeds) if tap_speeds else 0

        # Save data to CSV
        csv_file_path = 'finger_tap_data.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            fieldnames = ['Tap Count', 'Time', 'Distance (pixels)', 'Distance (cm)', 'Formatted Distance',
                          'Start Position', 'Tap Duration', 'Tap Speed (cm/s)']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for row in tap_data:
                writer.writerow(row)

        # Add average speed to the output
        st.write(f"Average Tap Speed: {average_speed:.2f} cm/s")

        # Add a download button for the CSV file
        st.download_button(
            label="Download Finger Tap Data (CSV)",
            data=open(csv_file_path, 'rb'),
            file_name="finger_tap_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()