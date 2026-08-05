"""Captura de vídeo e validação mínima de qualidade."""

from rPPG.capture.captura import (
    capture_video,
    capturar_video,
    is_frame_quality_acceptable,
    quadro_tem_qualidade_aceitavel,
)

__all__ = [
    "capturar_video",
    "quadro_tem_qualidade_aceitavel",
    "capture_video",
    "is_frame_quality_acceptable",
]
