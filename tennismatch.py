import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Define the model path
model_path = 'best.pt'  # Replace with your actual .pt file path

# Attempt to load the custom YOLOv5 model
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

# Apply CSS for enhanced styling
st.markdown(
    """
    <style>
        body {
            background-color: black; /* Main background */
            color: #00ff99; /* Text color */
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #101010; /* Container background */
            border-radius: 10px;
            padding: 10px;
        }
        .stSidebar {
            background-color: #202020; /* Sidebar background */
            border-right: 1px solid #444;
        }
        .stButton>button {
            color: white;
            background-color: #008080; /* Button background */
            border: none;
            border-radius: 5px;
            padding: 8px 12px;
        }
        .stButton>button:hover {
            background-color: #00cc99; /* Button hover color */
        }
        .stProgress > div > div > div {
            background-color: #00ff99; /* Progress bar color */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00ff99; /* Heading color */
        }
        p {
            color: #cccccc; /* Paragraph text color */
        }
        .uploadedVideo {
            border: 2px dashed #00cc99;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit Sidebar for user instructions
st.sidebar.title("🚀 Tennis Tracking for Players and Ball")
st.sidebar.info(
    """
    ### Welcome to the Tennis Tracking App! 🎾
    - **Step 1**: Upload a tennis video file (e.g., MP4, AVI, MOV).
    - **Step 2**: Watch as the app tracks players and the ball.
    - **Step 3**: Download the processed video.
    """
)

# Main App Interface
st.title('🎾 **Tennis Tracking Application**')
st.write("Detect and track players in tennis videos in real-time. Please upload a video file below to get started!")

# File uploader for video input
uploaded_video = st.file_uploader("Upload Your Tennis Video 🎥", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.markdown('<div class="uploadedVideo">Video uploaded successfully! Processing will begin shortly.</div>', unsafe_allow_html=True)

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("⏳ **Processing video... Please wait.**")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write frame to output video file
        out.write(frame)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate in display
        time.sleep(1 / fps)

    # Release video resources
    cap.release()
    out.release()

    st.success("🎉 **Video processing complete!**")

    # Provide download button for the processed video
    st.write("📥 **Download Your Processed Video:**")
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="⬇ **Download Processed Video**",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
