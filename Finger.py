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

# Define Streamlit app
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

        cap.release()
        cv2.destroyAllWindows()

        # Calculate the time of each individual tap and the speed
        tap_durations = []
        tap_speeds = []  # List to store speed of each tap
        for i in range(len(tap_timestamps) - 1):
            duration = tap_timestamps[i + 1] - tap_timestamps[i]
            tap_durations.append(duration)
            # Calculate speed (distance_cm/duration)
            speed = tap_data[i]['Distance (cm)'] / duration
            tap_speeds.append(speed)
            tap_data[i]['Speed (cm/s)'] = speed

        # Save data to CSV
        csv_file_path = 'finger_tap_data.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            fieldnames = ['Tap Count', 'Time', 'Distance (pixels)', 'Distance (cm)', 'Formatted Distance', 'Start Position', 'Tap Duration', 'Speed (cm/s)']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for i, row in enumerate(tap_data):
                row['Tap Duration'] = tap_durations[i] if i < len(tap_durations) else None
                writer.writerow(row)

        # Calculate and display the average of distance, time, and speed
        avg_distance = np.mean([tap['Distance (cm)'] for tap in tap_data])
        avg_time = np.mean(tap_durations)
        avg_speed = np.mean(tap_speeds)

        st.write(f"Average Distance: {avg_distance:.2f} cm")
        st.write(f"Average Time per Tap: {avg_time:.2f} s")
        st.write(f"Average Speed: {avg_speed:.2f} cm/s")

        # Add a download button for the CSV file
        st.download_button(
            label="Download Finger Tap Data (CSV)",
            data=open(csv_file_path, 'rb'),
            file_name="finger_tap_data.csv",
            mime="text/csv"
        )

        # Generate a PDF report
        pdf_file_path = 'finger_tap_report.pdf'
        generate_pdf_report(pdf_file_path, name, age, sex, tap_data, speeds_graph, avg_distance, avg_time, avg_speed)

        # Add a download button for the PDF report
        st.download_button(
            label="Download PDF Report",
            data=open(pdf_file_path, 'rb'),
            file_name="finger_tap_report.pdf",
            mime="application/pdf"
        )

def generate_pdf_report(pdf_file_path, name, age, sex, tap_data, speeds_graph, avg_distance, avg_time, avg_speed):
    c = canvas.Canvas(pdf_file_path, pagesize=letter)
    width, height = letter

    # Title and user information
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 40, "Finger Tap Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Name: {name}")
    c.drawString(50, height - 100, f"Age: {age}")
    c.drawString(50, height - 120, f"Sex: {sex}")

    # Add the average statistics
    c.drawString(50, height - 160, f"Average Distance: {avg_distance:.2f} cm")
    c.drawString(50, height - 180, f"Average Time per Tap: {avg_time:.2f} s")
    c.drawString(50, height - 200, f"Average Speed: {avg_speed:.2f} cm/s")

    # Add the graph
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        fig, ax = plt.subplots()
        ax.plot(speeds_graph, color='b')
        ax.set_title('Finger Tap Distance Over Time')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Distance (pixels)')
        fig.savefig(tmpfile.name)
        plt.close(fig)
        c.drawImage(tmpfile.name, 50, height - 400, width=500, height=200)

    # Add the detailed data table
    c.drawString(50, height - 440, "Detailed Data:")
    y = height - 460
    for tap in tap_data:
        if y < 50:
            c.showPage()
            y = height - 50
        tap_duration = tap.get('Tap Duration', 'N/A')
        tap_speed = tap.get('Speed (cm/s)', 'N/A')
        tap_duration_str = f"{tap_duration:.2f} s" if isinstance(tap_duration, (int, float)) else "N/A"
        tap_speed_str = f"{tap_speed:.2f} cm/s" if isinstance(tap_speed, (int, float)) else "N/A"
        c.drawString(50, y, f"Tap {tap['Tap Count']}: Distance = {tap['Distance (cm)']:.2f} cm, Duration = {tap_duration_str}, Speed = {tap_speed_str}")
        y -= 20

    c.save()

if __name__ == "__main__":
    main()
