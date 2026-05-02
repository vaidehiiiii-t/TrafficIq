from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


SUPPORTED_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")


@dataclass
class DetectionSummary:
    counts: dict[str, int]
    annotated_image: Image.Image
    detections: list[dict[str, Any]]


@dataclass
class VideoAnalysisSummary:
    counts: dict[str, int]
    preview_frames: list[Image.Image]
    frame_results: list[dict[str, Any]]
    frames_analyzed: int


class VehicleDetector:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)

    def detect(self, image_bytes: bytes, confidence: float = 0.25) -> DetectionSummary:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        return self.detect_array(image_array, confidence=confidence)

    def detect_array(self, image_array: np.ndarray, confidence: float = 0.25) -> DetectionSummary:
        image_rgb = self._ensure_rgb(image_array)
        results = self.model.predict(
            source=image_rgb,
            conf=confidence,
            verbose=False,
        )
        result = results[0]
        names = result.names
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        detections: list[dict[str, Any]] = []
        counts = {label: 0 for label in SUPPORTED_VEHICLE_CLASSES}

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                confidence_score = float(box.conf.item())
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "label": label,
                        "confidence": round(confidence_score, 3),
                        "bbox": [round(value, 1) for value in xyxy],
                    }
                )
                if label in counts:
                    counts[label] += 1

        return DetectionSummary(
            counts=counts,
            annotated_image=Image.fromarray(annotated_rgb),
            detections=detections,
        )

    def analyze_video(
        self,
        video_bytes: bytes,
        confidence: float = 0.25,
        frame_stride: int = 15,
        max_frames: int = 24,
    ) -> VideoAnalysisSummary:
        temp_path = self._write_temp_video(video_bytes)
        try:
            return self._analyze_capture(
                cv2.VideoCapture(str(temp_path)),
                confidence=confidence,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def analyze_stream(
        self,
        stream_source: str,
        confidence: float = 0.25,
        frame_stride: int = 15,
        max_frames: int = 24,
    ) -> VideoAnalysisSummary:
        capture = cv2.VideoCapture(stream_source)
        return self._analyze_capture(
            capture,
            confidence=confidence,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )

    def _analyze_capture(
        self,
        capture: cv2.VideoCapture,
        confidence: float,
        frame_stride: int,
        max_frames: int,
    ) -> VideoAnalysisSummary:
        counts = {label: 0 for label in SUPPORTED_VEHICLE_CLASSES}
        preview_frames: list[Image.Image] = []
        frame_results: list[dict[str, Any]] = []
        frames_analyzed = 0
        try:
            frame_index = 0
            while capture.isOpened() and frames_analyzed < max_frames:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % max(1, frame_stride) != 0:
                    frame_index += 1
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                summary = self.detect_array(rgb_frame, confidence=confidence)
                frames_analyzed += 1
                frame_index += 1

                for label, value in summary.counts.items():
                    counts[label] = max(counts[label], value)

                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "counts": summary.counts,
                        "detections": len(summary.detections),
                    }
                )
                if len(preview_frames) < 4:
                    preview_frames.append(summary.annotated_image)

            capture.release()
        finally:
            capture.release()

        return VideoAnalysisSummary(
            counts=counts,
            preview_frames=preview_frames,
            frame_results=frame_results,
            frames_analyzed=frames_analyzed,
        )

    @staticmethod
    def _ensure_rgb(image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 2:
            return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        if image_array.shape[2] == 4:
            return cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        return image_array

    @staticmethod
    def _write_temp_video(video_bytes: bytes) -> Path:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            return Path(temp_file.name)
