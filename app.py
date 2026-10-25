"""YOLO Object Detection Studio - Streamlit front end.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from yolostudio import counting, hosting, inference, live, models

st.set_page_config(page_title="YOLO Object Detection Studio", page_icon=":mag:", layout="wide")

HOSTED = hosting.is_hosted()


@dataclass
class Settings:
    confidence: float
    allowed: set[str] | None
    roi: counting.Roi
    show_centroids: bool


@st.cache_resource(show_spinner="Loading model...")
def cached_model(weights: str):
    return inference.load_model(weights)


def counts_table(total: Counter, in_roi: Counter) -> list[dict]:
    return [{"class": label, "count": total.get(label, 0), "in ROI": in_roi.get(label, 0)} for label in sorted(set(total) | set(in_roi))]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_image_result(model, image_bgr, settings: Settings, name: str) -> None:
    height, width = image_bgr.shape[:2]
    detections = inference.detect(model, image_bgr, settings.confidence, allowed=settings.allowed)
    total = counting.count_labels(detections)
    in_roi = counting.count_in_roi(detections, settings.roi, width, height)
    left, right = st.columns([3, 2])
    with left:
        st.image(inference.annotate(image_bgr, detections, settings.roi, settings.show_centroids), caption="Detections (green box = region of interest)")
    with right:
        m1, m2 = st.columns(2)
        m1.metric("Objects found", sum(total.values()))
        m2.metric("Inside ROI", sum(in_roi.values()))
        if detections:
            st.dataframe(counts_table(total, in_roi), hide_index=True)
            st.download_button("Download CSV", counting.build_csv(total, in_roi), f"{name}_{timestamp()}.csv", "text/csv")
            with st.expander("Every detection"):
                st.dataframe(
                    [{"class": d.label, "confidence": round(d.confidence, 3), "x1": int(d.x1), "y1": int(d.y1), "x2": int(d.x2), "y2": int(d.y2)} for d in detections],
                    hide_index=True,
                )
        else:
            st.info("Nothing detected. Try lowering the confidence threshold or choosing another model.")


def process_video(path: str, weights: str, settings: Settings, every_nth: int, max_frames: int, track: bool) -> None:
    import cv2

    model = inference.load_model(weights) if track else cached_model(weights)  # a fresh model resets tracker state
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        st.error("The video could not be opened.")
        return
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    wanted = min(max_frames, max(1, total_frames // every_nth))
    agg = counting.Aggregator()
    frame_slot, bar = st.empty(), st.progress(0.0, text="Processing video...")
    index = 0
    try:
        while agg.frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if (index - 1) % every_nth:
                continue
            height, width = frame.shape[:2]
            detections = inference.detect(model, frame, settings.confidence, track, settings.allowed)
            agg.add_frame(detections, settings.roi, width, height)
            frame_slot.image(inference.annotate(frame, detections, settings.roi, settings.show_centroids))
            bar.progress(min(1.0, agg.frames / wanted), text=f"Processed {agg.frames} frames")
    finally:
        capture.release()
    bar.empty()
    if agg.frames == 0:
        st.warning("No frames could be read from the video.")
        return
    st.success(f"Processed {agg.frames} frames.")
    st.dataframe(
        [
            {
                "class": label,
                "peak per frame": agg.peak.get(label, 0),
                "peak in ROI": agg.peak_roi.get(label, 0),
                "total detections": agg.detections.get(label, 0),
                **({"unique tracks": len(agg.unique_tracks.get(label, ()))} if track else {}),
            }
            for label in agg.labels()
        ],
        hide_index=True,
    )
    st.caption("'Total detections' counts an object once per frame it appears in. Turn on tracking for unique-object counts.")
    st.download_button("Download CSV", counting.build_video_csv(agg), f"video_{timestamp()}.csv", "text/csv")


def show_live_counts(slot_total, slot_roi, slot_frames, session: live.LiveSession) -> None:
    total, in_roi, frames = session.snapshot()
    slot_total.metric("Objects found", sum(total.values()))
    slot_roi.metric("Inside ROI", sum(in_roi.values()))
    slot_frames.caption(f"Frames analysed: {frames}")


def live_download(session: live.LiveSession) -> None:
    if session.aggregator.frames:
        st.download_button(
            "Download session CSV", counting.build_video_csv(session.aggregator), f"live_{timestamp()}.csv", "text/csv", key="live_csv"
        )
        st.caption("'Total detections' counts an object once per frame it appears in.")


def local_webcam(session: live.LiveSession) -> None:
    st.caption("Reads the camera of the computer running this app, so use it when you run the app on your own machine.")
    index = st.number_input("Camera index", 0, 10, 0, help="0 is the default webcam.")
    start, stop = st.columns(2)
    if start.button("Start webcam", type="primary"):
        session.reset()
        st.session_state.live_run = True
    if stop.button("Stop webcam"):
        st.session_state.live_run = False
    frame_slot = st.empty()
    m1, m2, m3 = st.columns(3)
    slot_total, slot_roi, slot_frames = m1.empty(), m2.empty(), m3.empty()
    if not st.session_state.get("live_run"):
        show_live_counts(slot_total, slot_roi, slot_frames, session)
        live_download(session)
        return
    import cv2

    capture = cv2.VideoCapture(int(index))
    if not capture.isOpened():
        st.session_state.live_run = False
        st.error("Could not open a camera on the machine running the app. If the app is hosted online, use the browser camera instead.")
        return
    try:
        while st.session_state.get("live_run"):
            ok, frame = capture.read()
            if not ok:
                st.session_state.live_run = False
                st.error("The camera stopped sending frames.")
                break
            frame_slot.image(session.process(frame), channels="BGR", use_container_width=True)
            show_live_counts(slot_total, slot_roi, slot_frames, session)
    finally:
        capture.release()


def browser_camera(session: live.LiveSession) -> None:
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.warning("The browser camera needs the optional packages: pip install streamlit-webrtc")
        return
    import av

    st.caption("Streams your browser's camera to the app, so it also works on a hosted deployment. Press START and allow camera access.")

    def on_frame(frame):
        return av.VideoFrame.from_ndarray(session.process(frame.to_ndarray(format="bgr24")), format="bgr24")

    ctx = webrtc_streamer(
        key="live-browser",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=on_frame,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    m1, m2, m3 = st.columns(3)
    slot_total, slot_roi, slot_frames = m1.empty(), m2.empty(), m3.empty()
    if ctx.state.playing:
        while ctx.state.playing:
            show_live_counts(slot_total, slot_roi, slot_frames, session)
            time.sleep(0.5)
    show_live_counts(slot_total, slot_roi, slot_frames, session)
    live_download(session)


# --------------------------------------------------------------------------- layout
st.title("YOLO Object Detection Studio")
st.caption("Detect and count objects in images, video and live camera, restrict counting to a region of interest, and export the counts.")

with st.sidebar:
    st.header("Model")
    keys = [m.key for m in models.MODELS]
    spec = models.get_model_spec(
        st.selectbox(
            "Model",
            keys,
            format_func=lambda k: models.get_model_spec(k).title,
        )
    )
    st.caption(spec.description)
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    only = st.text_input("Only these classes (comma-separated, optional)", placeholder="person, car")
    st.header("Region of interest")
    x1, x2 = st.slider("Horizontal (% of width)", 0, 100, (15, 85))
    y1, y2 = st.slider("Vertical (% of height)", 0, 100, (15, 85))
    show_centroids = st.checkbox("Mark object centres inside the ROI", value=True)

roi_error = None
try:
    roi = counting.Roi(x1 / 100, y1 / 100, x2 / 100, y2 / 100)
except ValueError as exc:
    roi, roi_error = counting.Roi(), str(exc)
if roi_error:
    st.sidebar.error(roi_error)

settings = Settings(
    confidence=confidence,
    allowed={c.strip() for c in only.split(",") if c.strip()} or None,
    roi=roi,
    show_centroids=show_centroids,
)

tab_image, tab_video, tab_live, tab_camera = st.tabs(["Image", "Video", "Live camera", "Snapshot"])

with tab_image:
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image")
    if upload is None:
        st.info("Upload an image to detect objects in it.")
    else:
        try:
            show_image_result(cached_model(spec.weights), inference.decode_image(upload.getvalue()), settings, "image")
        except ValueError as exc:
            st.error(str(exc))

with tab_video:
    clip = st.file_uploader("Choose a video", type=["mp4", "mov", "avi"], key="video")
    c1, c2, c3 = st.columns(3)
    every_nth = c1.number_input("Process every Nth frame", 1, 30, 3)
    max_frames = c2.number_input("Maximum frames to process", 10, 2000, 300, step=10)
    track = c3.checkbox("Track objects (unique counts)", value=False)
    if clip is None:
        st.info("Upload a short video, then press the button.")
    elif st.button("Process video", type="primary"):
        suffix = os.path.splitext(clip.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(clip.getvalue())
        try:
            process_video(handle.name, spec.weights, settings, int(every_nth), int(max_frames), track)
        finally:
            os.unlink(handle.name)

with tab_live:
    if "live_session" not in st.session_state:
        st.session_state.live_session = live.LiveSession()
    live_session: live.LiveSession = st.session_state.live_session
    if roi_error:
        st.error("Fix the region of interest in the sidebar first.")
    else:
        live_session.configure(spec.weights, settings.confidence, settings.allowed, settings.roi, settings.show_centroids)
        if HOSTED:  # the server has no webcam, so only the visitor's browser camera makes sense
            source = "Browser camera"
        else:
            source = st.radio("Camera source", ["This computer's webcam", "Browser camera"], horizontal=True, key="live_source")
        if source == "Browser camera":
            browser_camera(live_session)
        else:
            local_webcam(live_session)

with tab_camera:
    st.caption("Takes one picture with your browser's camera and analyses it.")
    shot = st.camera_input("Take a picture")
    if shot is not None:
        show_image_result(cached_model(spec.weights), inference.decode_image(shot.getvalue()), settings, "camera")
