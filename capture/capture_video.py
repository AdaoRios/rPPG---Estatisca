"""Webcam preview, continuous capture, and real-time quality feedback."""

from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import time

import cv2

from rPPG.capture.quality_check import (
    FaceFramingQualityChecker,
    MovementQualityChecker,
)
from rPPG.capture.lighting_quality import LightingQualityChecker
from rPPG.config import (
    FACE_MAX_CENTER_OFFSET_X,
    FACE_MAX_CENTER_OFFSET_Y,
    FACE_MAX_HEIGHT_RATIO,
    FACE_MAX_WIDTH_RATIO,
    FACE_MIN_HEIGHT_RATIO,
    FACE_MIN_WIDTH_RATIO,
    LIGHTING_BRIGHT_PIXEL_CHANNEL,
    LIGHTING_DARK_PIXEL_LUMINANCE,
    LIGHTING_LUMA_WEIGHTS,
    LIGHTING_UNIFORMITY_GRID_COLUMNS,
    LIGHTING_UNIFORMITY_GRID_ROWS,
    MODEL_PATH,
    MOVEMENT_THRESHOLD,
    MOVEMENT_WINDOW_SIZE,
    READY_STABLE_FRAMES,
)
from rPPG.roi.face_detection import FaceDetector


class CaptureState(Enum):
    """User-visible stages of webcam capture."""

    PREVIEW = auto()
    CAPTURING = auto()


def _draw_lines(frame, lines, color=(0, 255, 0)):
    """Draw a compact OpenCV overlay using ASCII-safe status text."""
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (20, 35 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)


def _draw_face_bbox(frame, bbox, color):
    """Draw the landmark-derived bounding box for framing calibration."""
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


def _format_ratio(value):
    return "--" if value is None else f"{value * 100:.1f}%"


def _format_metric(value):
    return "--" if value is None else f"{value:.3f}"


def capture_video(camera_index=0, duration_s=30.0, output_dir=None):
    """Preview first, then record every camera frame after ENTER is pressed.

    Quality components only provide feedback. They never filter, pause, or
    restart the continuous MP4 once ``CAPTURING`` begins.
    """
    output_dir = (Path(output_dir) if output_dir else
                  Path(__file__).resolve().parents[1] / "data" / "captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"video_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector(MODEL_PATH)
    movement_checker = MovementQualityChecker(
        MOVEMENT_THRESHOLD, MOVEMENT_WINDOW_SIZE, READY_STABLE_FRAMES
    )
    framing_checker = FaceFramingQualityChecker(
        FACE_MIN_WIDTH_RATIO,
        FACE_MAX_WIDTH_RATIO,
        FACE_MIN_HEIGHT_RATIO,
        FACE_MAX_HEIGHT_RATIO,
        FACE_MAX_CENTER_OFFSET_X,
        FACE_MAX_CENTER_OFFSET_Y,
    )
    lighting_checker = LightingQualityChecker(
        LIGHTING_LUMA_WEIGHTS,
        LIGHTING_DARK_PIXEL_LUMINANCE,
        LIGHTING_BRIGHT_PIXEL_CHANNEL,
        LIGHTING_UNIFORMITY_GRID_ROWS,
        LIGHTING_UNIFORMITY_GRID_COLUMNS,
    )
    writer = None
    captured_frames = 0
    state = CaptureState.PREVIEW
    capture_start_time = None
    session_start_time = time.monotonic()
    try:
        print("Preview aberto. Pressione ENTER para iniciar a captura.")
        while True:
            success, frame = capture.read()
            if not success:
                break
            elapsed_ms = int((time.monotonic() - session_start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame, elapsed_ms)
            movement = movement_checker.update(landmarks)
            framing = framing_checker.evaluate(landmarks, frame.shape)
            lighting = lighting_checker.evaluate(rgb_frame, framing.bbox)

            metric_text = "--" if movement.movement_metric is None else f"{movement.movement_metric:.4f}"
            lines = [
                f"Movimento: {metric_text}",
                f"Threshold: {movement.movement_threshold:.4f}",
                f"Face width: {_format_ratio(framing.face_width_ratio)}",
                f"Face height: {_format_ratio(framing.face_height_ratio)}",
                f"Center X: {_format_ratio(framing.face_center_x)}",
                f"Center Y: {_format_ratio(framing.face_center_y)}",
                "Lighting",
                f"Mean luminance: {_format_metric(lighting.mean_luminance)}",
                f"Dark ratio: {_format_ratio(lighting.dark_pixel_ratio)}",
                f"Bright ratio: {_format_ratio(lighting.bright_pixel_ratio)}",
                f"Uniformity: {_format_metric(lighting.illumination_uniformity)}",
            ]
            color = (0, 255, 0)
            if state is CaptureState.PREVIEW:
                lines = [
                    "PREVIEW",
                    "Posicione-se para a captura",
                    "Pressione ENTER para iniciar",
                    "Pressione Q para sair",
                    *lines,
                ]
                color = (255, 255, 0)
            else:
                movement_status = "OK" if movement.is_stable else "MOVIMENTO DETECTADO"
                lines = [
                    "CAPTURANDO",
                    f"Movement: {movement_status}",
                    f"Framing: {framing.message}",
                    *lines,
                ]
                if not framing.face_detected:
                    lines.append("Posicione seu rosto na camera")
                    color = (0, 0, 255)
                elif not framing.framing_ok:
                    color = (0, 165, 255)
                elif not movement.is_stable:
                    lines.append("Mantenha o rosto imovel")
                    color = (0, 165, 255)

            capture_finished = False
            if state is CaptureState.CAPTURING:
                # This is deliberately unconditional: quality never drops frames.
                writer.write(frame)
                captured_frames += 1
                remaining = max(0.0, duration_s - (time.monotonic() - capture_start_time))
                lines.append(f"Tempo restante: {remaining:.1f}s")
                capture_finished = remaining <= 0.0

            _draw_face_bbox(frame, framing.bbox, color)
            _draw_lines(frame, lines, color)
            cv2.imshow("rPPG - Captura", frame)
            if capture_finished:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if state is CaptureState.PREVIEW and key in (10, 13):
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Nao foi possivel criar o arquivo de video.")
                capture_start_time = time.monotonic()
                state = CaptureState.CAPTURING
                print("Captura iniciada. Qualidade sera monitorada durante a gravacao.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        cv2.destroyAllWindows()

    if captured_frames < 2:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("Captura cancelada antes de gravar frames suficientes.")
    print(f"Captura concluida: {captured_frames} frames gravados em {output_path}")
    return str(output_path)
