
import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO

st.set_page_config(
    page_title="Vehicle Detection and Speed Estimation",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Detection and Speed Estimation")
st.write("YOLO-powered vehicle detection and approximate speed estimation.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

uploaded_video = st.file_uploader(
    "Upload a road traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)

st.info(
    "For approximate speed estimation, enter the number of pixels "
    "per meter measured in your video. Accurate results require "
    "camera calibration."
)

pixels_per_meter = st.number_input(
    "Pixels per meter",
    min_value=1.0,
    max_value=10000.0,
    value=20.0,
    step=1.0
)

if uploaded_video is not None:
    if st.button("▶ Start Vehicle Detection", type="primary"):
        input_path = None
        output_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_input:
                temp_input.write(uploaded_video.getbuffer())
                input_path = temp_input.name

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Could not open this video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or width <= 0 or height <= 0:
                cap.release()
                st.error("The video has invalid dimensions or no frames.")
                st.stop()

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as temp_output:
                output_path = temp_output.name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                cap.release()
                st.error("Could not create the output video.")
                st.stop()

            model = load_model()
            previous_positions = {}
            speeds = {}
            vehicle_ids = set()

            progress = st.progress(0)
            status = st.empty()
            frame_number = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_number += 1

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[2, 3, 5, 7],
                    imgsz=640,
                    conf=0.35,
                    verbose=False
                )

                result = results[0]
                annotated = result.plot()

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    labels = result.boxes.cls.int().cpu().tolist()

                    for box, track_id, class_id in zip(boxes, ids, labels):
                        x1, y1, x2, y2 = box
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        vehicle_ids.add(track_id)

                        if track_id in previous_positions:
                            old_x, old_y, old_frame = previous_positions[track_id]
                            elapsed = (frame_number - old_frame) / fps

                            if elapsed > 0:
                                pixel_distance = (
                                    (center_x - old_x) ** 2
                                    + (center_y - old_y) ** 2
                                ) ** 0.5

                                speed_kmh = (
                                    pixel_distance / pixels_per_meter
                                ) / elapsed * 3.6

                                # Ignore implausible jumps and smooth estimates.
                                if speed_kmh <= 200:
                                    old_speed = speeds.get(track_id, speed_kmh)
                                    speeds[track_id] = (
                                        0.7 * old_speed + 0.3 * speed_kmh
                                    )

                        previous_positions[track_id] = (
                            center_x, center_y, frame_number
                        )

                        speed_text = (
                            f"{speeds[track_id]:.1f} km/h"
                            if track_id in speeds
                            else "Calculating..."
                        )

                        cv2.putText(
                            annotated,
                            f"ID {track_id}: {speed_text}",
                            (int(x1), max(25, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                writer.write(annotated)
                progress.progress(min(frame_number / total_frames, 1.0))
                status.write(f"Processing frame {frame_number}/{total_frames}")

            cap.release()
            writer.release()

            st.success("Video processing completed!")
            st.metric("Unique vehicle tracks detected", len(vehicle_ids))

            with open(output_path, "rb") as video_file:
                st.video(video_file.read())

            with open(output_path, "rb") as video_file:
                st.download_button(
                    "⬇ Download Processed Video",
                    data=video_file.read(),
                    file_name="vehicle_detection_output.mp4",
                    mime="video/mp4"
                )

            st.warning(
                "Speed values are estimates, not certified measurements. "
                "They depend on camera calibration, perspective, tracking "
                "quality, and the pixels-per-meter setting."
            )

        except Exception as error:
            st.error(f"Processing failed: {error}")

        finally:
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
else:
    st.caption("Upload a video to begin.")
