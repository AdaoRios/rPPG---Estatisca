"""Shared rPPG configuration."""

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "face_landmarker.task"
DEBUG_COMPARE_ALGORITHMS = True

METHOD_WEIGHTS = {
    "chrom": 0.0,
    "pos": 1.0,
    "green": 0.0,
}

ROI_WEIGHTS = {
    "testa": 0.4,
    "bochecha_esquerda": 0.3,
    "bochecha_direita": 0.3,
}

ROI_POINTS = {
    "testa": [67, 109, 10, 338, 297, 332, 284, 251, 301, 71],
    "bochecha_esquerda": [50, 187, 205, 36, 142, 126, 209, 49, 129, 203],
    "bochecha_direita": [280, 411, 425, 266, 371, 355, 429, 279, 358, 423],
}
