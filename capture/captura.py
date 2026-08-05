"""Captura de vídeo por webcam e validação mínima de cada quadro."""

from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from rPPG.config import MODEL_PATH
from rPPG.roi.facial import DetectorFace


def quadro_tem_qualidade_aceitavel(rgb_frame, landmarks):
    """Aceita quadros com face detectada e brilho médio não extremo."""
    if landmarks is None:
        return False
    brightness = float(np.mean(rgb_frame))
    return 20.0 <= brightness <= 235.0


def capturar_video(camera_index=0, duration_s=30.0, output_dir=None):
    """Grava quadros válidos da webcam em MP4 e retorna o caminho criado."""
    output_dir = (
        Path(output_dir)
        if output_dir
        else Path(__file__).resolve().parents[1] / "data" / "captures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"video_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = DetectorFace(MODEL_PATH)
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
            landmarks = detector.detectar(rgb_frame, int(elapsed * 1000))

            # O arquivo só recebe quadros que passaram pela validação da captura.
            if quadro_tem_qualidade_aceitavel(rgb_frame, landmarks):
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                writer.write(frame)
                valid_frames += 1

            remaining = max(0.0, duration_s - elapsed)
            cv2.putText(
                frame,
                f"Capturando... {remaining:0.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("rPPG - Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.fechar()
        cv2.destroyAllWindows()

    if valid_frames < 2:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("Poucos frames válidos capturados. Verifique iluminação e posicionamento.")
    print(f"Captura concluída: {valid_frames} frames válidos em {output_path}")
    return str(output_path)


# Aliases preservam os nomes técnicos existentes para consumidores externos.
capture_video = capturar_video
is_frame_quality_acceptable = quadro_tem_qualidade_aceitavel

__all__ = [
    "capturar_video",
    "quadro_tem_qualidade_aceitavel",
    "capture_video",
    "is_frame_quality_acceptable",
]
