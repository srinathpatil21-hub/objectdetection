
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# -------------------- PAGE SETTINGS --------------------

st.set_page_config(
    page_title="YOLO Vision - Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 YOLO Vision")
st.subheader("Advanced Object Detection Suite")
st.write(
    "Upload a video, choose a YOLO model and object classes, "
    "then download your video with detected objects."
)


# -------------------- MODEL LOADING --------------------

@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


# -------------------- MODEL SELECTION --------------------

st.sidebar.header("⚙️ Detection Settings")

model_size = st.sidebar.selectbox(
    "Select YOLO Model Size",
    ["Nano (n)", "Small (s)", "Medium (m)", "Large (l)", "Extra Large (x)"]
)

size_map = {
    "Nano (n)": "n",
    "Small (s)": "s",
    "Medium (m)": "m",
    "Large (l)": "l",
    "Extra Large (x)": "x"
}

model_name = f"yolov8{size_map[model_size]}.pt"

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05
)

st.sidebar.info(
    f"Selected model: {model_name}\n\n"
    "Larger models may be slower and require more memory."
)


# -------------------- VIDEO UPLOAD --------------------

uploaded_video = st.file_uploader(
    "📁 Upload a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🔍 Load Model and Detect Objects", type="primary"):

        input_path = None
        output_path = None
        cap = None

        try:
            # Save uploaded video temporarily
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as input_file:
                input_file.write(uploaded_video.getbuffer())
                input_path = input_file.name

            # Load selected model
            with st.spinner(
                f"Loading {model_name}. The first download may take a while..."
            ):
                model = load_model(model_name)

            # Open video
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error(
                    "Could not open this video. Try another MP4 or AVI file."
                )
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0 or width <= 0 or height <= 0:
                st.error("The video has invalid dimensions or frame rate.")
                st.stop()

            # Prepare output video
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as output_file:
                output_path = output_file.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )

            if not writer.isOpened():
                st.error("Could not create the output video.")
                st.stop()

            # Get class names from the model
            class_names = model.names

            if isinstance(class_names, dict):
                class_ids = sorted(class_names.keys())
                class_labels = [
                    f"{i}: {class_names[i]}" for i in class_ids
                ]
            else:
                class_ids = list(range(len(class_names)))
                class_labels = [
                    f"{i}: {class_names[i]}" for i in class_ids
                ]

            selected_labels = st.multiselect(
                "🎯 Select object classes to detect",
                options=class_labels,
                default=[
                    label for label in class_labels
                    if label.split(": ", 1)[-1] in [
                        "person", "bicycle", "car",
                        "motorcycle", "bus", "truck"
                    ]
                ],
                help="Leave empty to detect all classes."
            )

            selected_ids = [
                int(label.split(": ", 1)[0])
                for label in selected_labels
            ]

            # Process video
            progress = st.progress(0)
            status = st.empty()

            processed_frames = 0
            detected_objects = 0
            start_time = time.time()

            while True:
                success, frame = cap.read()

                if not success:
                    break

                # Empty class selection means all classes
                classes_to_use = selected_ids if selected_ids else None

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    classes=classes_to_use,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                writer.write(annotated_frame)

                if results[0].boxes is not None:
                    detected_objects += len(results[0].boxes)

                processed_frames += 1

                if total_frames > 0:
                    progress.progress(
                        min(processed_frames / total_frames, 1.0)
                    )

                if processed_frames % 10 == 0:
                    status.write(
                        f"Processed {processed_frames} / "
                        f"{total_frames if total_frames > 0 else '?'} frames"
                    )

            cap.release()
            cap = None
            writer.release()

            elapsed = time.time() - start_time

            progress.progress(1.0)
            status.success("Video processing completed!")

            st.success("✅ Object detection completed successfully!")

            col1, col2, col3 = st.columns(3)

            col1.metric("Frames Processed", processed_frames)
            col2.metric("Objects Detected", detected_objects)
            col3.metric("Processing Time", f"{elapsed:.1f} sec")

            # Display and download result
            st.subheader("🎬 Processed Video")

            with open(output_path, "rb") as result_file:
                video_bytes = result_file.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_bytes,
                file_name="yolo_detected_video.mp4",
                mime="video/mp4"
            )

        except Exception as error:
            st.error(f"Something went wrong: {error}")
            st.info(
                "Try a smaller model, a shorter video, or a supported video format."
            )

        finally:
            if cap is not None:
                cap.release()

            for path in [input_path, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

else:
    st.info("👆 Upload a video above to begin object detection.")

st.divider()
st.caption("YOLO Vision | Powered by Ultralytics YOLO and OpenCV")
