"""Algoritmos de extração e fusão de sinais rPPG.

Este arquivo mantém o diretório como um pacote explícito e reúne sua API
principal, sem implementar cálculos próprios.
"""

from rPPG.extractors.algorithms import chrom_algorithm, green_algorithm, pos_algorithm
from rPPG.extractors.combination import combine_roi_and_methods

__all__ = [
    "chrom_algorithm",
    "pos_algorithm",
    "green_algorithm",
    "combine_roi_and_methods",
]
