from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trafficiq.detection import VehicleDetector
from trafficiq.legacy_bridge import legacy_source_status, load_legacy_signal_config
from trafficiq.traffic_logic import SignalTimingConfig, build_signal_plan, normalize_vehicle_counts, simulate_signal_cycles


DIRECTIONS = ("North", "East", "South", "West")
DEFAULT_MODEL = "yolov8n.pt"
TRAFFIC_CLASSES = ("car", "motorcycle", "bus", "truck", "rickshaw")
LEGACY_CONFIG = load_legacy_signal_config()
LEGACY_STATUS = legacy_source_status()
LEGACY_TEST_IMAGES_DIR = Path(__file__).resolve().parent / "Code" / "YOLO" / "darkflow" / "test_images"


st.set_page_config(
    page_title="TrafficIQ",
    page_icon="🚦",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_detector(model_path: str) -> VehicleDetector:
    return VehicleDetector(model_path)


def empty_counts() -> dict[str, int]:
    return {label: 0 for label in TRAFFIC_CLASSES}


def load_legacy_test_images() -> OrderedDict[str, bytes]:
    image_paths = sorted(
        path for path in LEGACY_TEST_IMAGES_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    images: OrderedDict[str, bytes] = OrderedDict()
    for index, direction in enumerate(DIRECTIONS):
        if not image_paths:
            break
        image_path = image_paths[index % len(image_paths)]
        images[direction] = image_path.read_bytes()
    return images


def ensure_all_directions(counts_by_direction: dict[str, dict[str, int]]) -> OrderedDict[str, dict[str, int]]:
    merged: OrderedDict[str, dict[str, int]] = OrderedDict()
    for direction in DIRECTIONS:
        merged[direction] = normalize_vehicle_counts(counts_by_direction.get(direction, empty_counts()))
    return merged


def count_from_uploaded_images(
    uploaded_images: OrderedDict[str, bytes],
    detector: VehicleDetector,
    confidence: float,
) -> tuple[OrderedDict[str, dict[str, int]], dict[str, Any]]:
    direction_counts: dict[str, dict[str, int]] = {}
    annotated_images: dict[str, Any] = {}

    for direction, image_bytes in uploaded_images.items():
        summary = detector.detect(image_bytes, confidence=confidence)
        direction_counts[direction] = normalize_vehicle_counts(summary.counts)
        annotated_images[direction] = summary.annotated_image

    return ensure_all_directions(direction_counts), annotated_images


def count_from_uploaded_videos(
    uploaded_videos: OrderedDict[str, bytes],
    detector: VehicleDetector,
    confidence: float,
    frame_stride: int,
    max_frames: int,
) -> tuple[OrderedDict[str, dict[str, int]], dict[str, list[Any]], dict[str, list[dict[str, Any]]]]:
    direction_counts: dict[str, dict[str, int]] = {}
    preview_frames: dict[str, list[Any]] = {}
    frame_results: dict[str, list[dict[str, Any]]] = {}

    for direction, video_bytes in uploaded_videos.items():
        summary = detector.analyze_video(
            video_bytes,
            confidence=confidence,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        direction_counts[direction] = normalize_vehicle_counts(summary.counts)
        preview_frames[direction] = summary.preview_frames
        frame_results[direction] = summary.frame_results

    return ensure_all_directions(direction_counts), preview_frames, frame_results


def count_from_live_streams(
    stream_sources: OrderedDict[str, str],
    detector: VehicleDetector,
    confidence: float,
    frame_stride: int,
    max_frames: int,
) -> tuple[OrderedDict[str, dict[str, int]], dict[str, list[Any]], dict[str, list[dict[str, Any]]]]:
    direction_counts: dict[str, dict[str, int]] = {}
    preview_frames: dict[str, list[Any]] = {}
    frame_results: dict[str, list[dict[str, Any]]] = {}

    for direction, stream_source in stream_sources.items():
        summary = detector.analyze_stream(
            stream_source,
            confidence=confidence,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        direction_counts[direction] = normalize_vehicle_counts(summary.counts)
        preview_frames[direction] = summary.preview_frames
        frame_results[direction] = summary.frame_results

    return ensure_all_directions(direction_counts), preview_frames, frame_results


def manual_adjustments_ui(prefix: str, direction: str, base_counts: dict[str, int]) -> dict[str, int]:
    st.caption(f"{direction} vehicle counts")
    cols = st.columns(len(TRAFFIC_CLASSES))
    updated: dict[str, int] = {}
    for index, label in enumerate(TRAFFIC_CLASSES):
        default_value = int(base_counts.get(label, 0))
        updated[label] = cols[index].number_input(
            f"{direction} {label}",
            min_value=0,
            value=default_value,
            step=1,
            key=f"{prefix}-{direction}-{label}",
        )
    return updated


def render_overview(plan: list[dict[str, object]], counts: OrderedDict[str, dict[str, int]]) -> None:
    schedule_df = pd.DataFrame(plan)
    counts_df = pd.DataFrame(counts).T.reset_index().rename(columns={"index": "direction"})

    total_detected = int(counts_df[list(TRAFFIC_CLASSES)].sum().sum())
    busiest_direction = schedule_df.sort_values("weighted_load", ascending=False).iloc[0]["direction"]
    cycle_time = int(schedule_df["green_time"].sum() + schedule_df["yellow_time"].sum())

    metric_cols = st.columns(3)
    metric_cols[0].metric("Detected vehicles", total_detected)
    metric_cols[1].metric("Busiest approach", str(busiest_direction))
    metric_cols[2].metric("Full cycle time", f"{cycle_time} sec")

    left, right = st.columns((1.2, 1))
    with left:
        st.subheader("Recommended signal plan")
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Vehicle counts by direction")
        st.bar_chart(
            counts_df.set_index("direction")[list(TRAFFIC_CLASSES)],
            use_container_width=True,
        )


def render_image_results(annotated_images: dict[str, Any], title: str) -> None:
    if not annotated_images:
        return
    st.subheader(title)
    preview_cols = st.columns(max(1, len(annotated_images)))
    for index, (direction, image) in enumerate(annotated_images.items()):
        preview_cols[index].image(image, caption=f"{direction} annotated view", use_container_width=True)


def render_video_results(preview_frames: dict[str, list[Any]], frame_results: dict[str, list[dict[str, Any]]]) -> None:
    if not preview_frames:
        return
    st.subheader("Video detection previews")
    for direction in DIRECTIONS:
        frames = preview_frames.get(direction, [])
        if not frames:
            continue
        st.markdown(f"**{direction} direction**")
        cols = st.columns(len(frames))
        for index, frame in enumerate(frames):
            cols[index].image(frame, caption=f"{direction} sampled frame {index + 1}", use_container_width=True)
        if frame_results.get(direction):
            st.dataframe(pd.DataFrame(frame_results[direction]), use_container_width=True, hide_index=True)


def render_timing_editor(prefix: str, detected_counts: OrderedDict[str, dict[str, int]], config: SignalTimingConfig) -> OrderedDict[str, dict[str, int]]:
    st.subheader("Refine counts")
    st.write(
        "Pretrained COCO models detect cars, motorcycles, buses, and trucks. "
        "Rickshaw counts can be added or corrected manually before calculating the final timings."
    )
    final_counts: OrderedDict[str, dict[str, int]] = OrderedDict()
    for direction in DIRECTIONS:
        final_counts[direction] = manual_adjustments_ui(prefix, direction, detected_counts[direction])
    plan = build_signal_plan(final_counts, config)
    render_overview(plan, final_counts)
    st.session_state["latest_counts"] = final_counts
    st.session_state["latest_plan"] = plan
    return final_counts


def render_simulation(config: SignalTimingConfig) -> None:
    st.subheader("Adaptive simulation")
    st.write(
        "This web simulation uses the same weighted vehicle-load concept as the rest of TrafficIQ. "
        "It models how timings change cycle by cycle without relying on a desktop game loop."
    )

    default_counts = st.session_state.get("latest_counts")
    if default_counts:
        st.info("Using the latest analyzed counts from the app. You can override them below.")
    else:
        default_counts = ensure_all_directions({})

    simulation_counts: OrderedDict[str, dict[str, int]] = OrderedDict()
    for direction in DIRECTIONS:
        simulation_counts[direction] = manual_adjustments_ui("simulation", direction, default_counts[direction])

    cycles = st.slider("Number of cycles to simulate", min_value=1, max_value=5, value=2)
    plan = build_signal_plan(simulation_counts, config)
    timeline = simulate_signal_cycles(simulation_counts, config, cycles=cycles)
    timeline_df = pd.DataFrame(timeline)

    render_overview(plan, simulation_counts)

    if not timeline_df.empty:
        st.subheader("Signal timeline")
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        st.subheader("Queue depletion trend")
        trend_df = timeline_df.pivot_table(
            index="global_second",
            columns="direction",
            values="remaining_weighted_load",
            aggfunc="last",
        )
        st.line_chart(trend_df, use_container_width=True)

    st.markdown(
        """
        How this simulation works:
        - The green time formula uses weighted vehicle load and lane capacity.
        - Each green second reduces the active direction's weighted queue load by lane-based service capacity.
        - The result is an analytical signal-cycle simulation designed for the web app.
        """
    )


def main() -> None:
    st.title("TrafficIQ")
    st.write(
        "TrafficIQ analyzes junction images and videos with YOLO, estimates directional load, "
        "and recommends adaptive signal timings through a deployable Streamlit interface."
    )
    st.caption(
        "The deployed app keeps the original signal-control formula by loading timing defaults and "
        "vehicle pass-time weights from the restored legacy `simulation.py` source."
    )

    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input("YOLO model", value=DEFAULT_MODEL)
        confidence = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
        lanes = st.slider("Lanes per direction", min_value=1, max_value=4, value=LEGACY_CONFIG.lanes)
        min_green = st.slider("Minimum green time", min_value=5, max_value=30, value=LEGACY_CONFIG.default_minimum)
        max_green = st.slider("Maximum green time", min_value=20, max_value=120, value=LEGACY_CONFIG.default_maximum)
        yellow_time = st.slider("Yellow time", min_value=3, max_value=10, value=LEGACY_CONFIG.default_yellow)
        frame_stride = st.slider("Video frame stride", min_value=1, max_value=60, value=15)
        max_frames = st.slider("Max sampled video frames", min_value=4, max_value=60, value=24)
        st.info(
            "Default deployment uses `yolov8n.pt`. Replace it with your own traffic model for better accuracy."
        )
        st.caption(f"Legacy core loaded from: `{LEGACY_STATUS['simulation_path']}`")

    config = SignalTimingConfig(
        min_green=min_green,
        max_green=max_green,
        yellow_time=yellow_time,
        lanes_per_direction=lanes,
    )

    try:
        detector = load_detector(model_path)
    except Exception as exc:
        st.error(f"Unable to load the YOLO model `{model_path}`. Details: {exc}")
        return

    image_tab, video_tab, simulation_tab = st.tabs(["Images", "Video + Camera", "Simulation"])

    with image_tab:
        st.subheader("Legacy test image analysis")
        st.write("This tab always uses the restored legacy test images and immediately shows the detected output.")
        uploaded_images = load_legacy_test_images()

        if not uploaded_images:
            st.error(f"No test images found in `{LEGACY_TEST_IMAGES_DIR}`.")
        else:
            st.caption(f"Using test images from `{LEGACY_TEST_IMAGES_DIR}`")
            with st.spinner("Running vehicle detection on legacy test images..."):
                detected_counts, annotated_images = count_from_uploaded_images(uploaded_images, detector, confidence)
            render_image_results(annotated_images, "Legacy test image outputs")
            render_timing_editor("image", detected_counts, config)

    with video_tab:
        st.subheader("Video and live CCTV analysis")
        st.write(
            "Upload direction-wise traffic videos or connect a live CCTV stream URL. "
            "The app samples frames, runs YOLO on them, and keeps the peak detected load per class."
        )
        video_cols = st.columns(4)
        uploaded_videos: OrderedDict[str, bytes] = OrderedDict()
        for index, direction in enumerate(DIRECTIONS):
            uploaded_file = video_cols[index].file_uploader(
                f"{direction} video",
                type=["mp4", "mov", "avi", "mkv"],
                key=f"video-{direction}",
            )
            if uploaded_file is not None:
                uploaded_videos[direction] = uploaded_file.getvalue()

        st.markdown("**Live CCTV stream inputs**")
        stream_cols = st.columns(2)
        stream_sources: OrderedDict[str, str] = OrderedDict()
        for index, direction in enumerate(DIRECTIONS):
            stream_value = stream_cols[index % 2].text_input(
                f"{direction} RTSP/HTTP stream",
                value="",
                placeholder="rtsp://... or http://...",
                key=f"stream-{direction}",
            ).strip()
            if stream_value:
                stream_sources[direction] = stream_value

        if uploaded_videos:
            with st.spinner("Sampling video frames and running detection..."):
                detected_counts, preview_frames, frame_results = count_from_uploaded_videos(
                    uploaded_videos,
                    detector,
                    confidence,
                    frame_stride,
                    max_frames,
                )
            render_video_results(preview_frames, frame_results)
            render_timing_editor("video", detected_counts, config)
        elif stream_sources:
            with st.spinner("Sampling live CCTV frames and running detection..."):
                detected_counts, preview_frames, frame_results = count_from_live_streams(
                    stream_sources,
                    detector,
                    confidence,
                    frame_stride,
                    max_frames,
                )
            render_video_results(preview_frames, frame_results)
            render_timing_editor("stream", detected_counts, config)
        else:
            st.info("Upload one or more approach videos or enter at least one live CCTV stream URL.")

    with simulation_tab:
        render_simulation(config)
    st.subheader("Deployment flow")
    st.markdown(
        """
        - Use the `Images` tab for static junction snapshots.
        - Use the `Video + Camera` tab for uploaded traffic videos or live CCTV stream URLs.
        - Use the `Simulation` tab to inspect cycle behavior before deploying a model configuration.
        """
    )


if __name__ == "__main__":
    main()
