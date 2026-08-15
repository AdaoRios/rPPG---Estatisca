"""Shared rPPG configuration."""
MODEL_PATH = "assets/face_landmarker.task"
DEBUG_COMPARE_ALGORITHMS = False

HR_LOW_HZ = 0.8    # 48 bpm
HR_HIGH_HZ = 3.0   # 180 bpm

METHOD_WEIGHTS = {
    "chrom": 0.3,
    "pos": 0.3,
    "ica": 0.3,
    "green": 0.1
}

ROI_WEIGHTS = {
    "testa": 0.4,
    "bochecha_esquerda": 0.3,
    "bochecha_direita": 0.3,
}

ROI_POINTS = {
    "testa": [109, 67, 103, 10, 332, 297, 338, 151, 9, 8],
    "bochecha_esquerda": [117, 118, 101, 123, 187, 207, 192, 214, 138, 135, 198, 50],
    "bochecha_direita": [346, 347, 330, 352, 411, 427, 416, 434, 367, 364, 418, 280],
}