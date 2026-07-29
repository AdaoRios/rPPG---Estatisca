"""Webcam capture pipeline that saves quality-checked MP4 video."""

from datetime import datetime
from pathlib import Path
import time

import cv2

from rPPG.capture.quality_check import is_frame_quality_acceptable
from rPPG.config import MODEL_PATH
from rPPG.roi.face_detection import FaceDetector


def capture_video(camera_index=0, duration_s=30.0, output_dir=None):
    """Capture quality-checked webcam frames into an MP4 and return its path."""
    output_dir = (Path(output_dir) if output_dir else
                  Path(__file__).resolve().parents[1] / "data" / "captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"video_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector(MODEL_PATH)
    writer = None
    valid_frames = 0
    start_time = time.time()
    try:
        print(f"Captura iniciada ({duration_s}s). Mantenha o rosto estável e bem iluminado.")
        while time.time() - start_time <= duration_s:
            success, frame = capture.read()
            if not success:
                break
            elapsed = time.time() - start_time
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame, int(elapsed * 1000))
            if is_frame_quality_acceptable(rgb_frame, landmarks):
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps, (width, height))
                writer.write(frame)
                valid_frames += 1
            remaining = max(0.0, duration_s - elapsed)
            cv2.putText(frame, f"Capturando... {remaining:0.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("rPPG - Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        cv2.destroyAllWindows()

    if valid_frames < 2:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("Poucos frames válidos capturados. Verifique iluminação e posicionamento.")
    print(f"Captura concluída: {valid_frames} frames válidos em {output_path}")
    return str(output_path)
