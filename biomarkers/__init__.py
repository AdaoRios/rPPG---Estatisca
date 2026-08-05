"""API de biomarcadores para análise rPPG.

Este pacote expõe funções didáticas para:
- estimar frequência cardíaca usando FFT,
- calcular variabilidade da frequência cardíaca (HRV),
- calcular métricas de qualidade de sinal,
- estimar taxa respiratória futura.
"""

from .heart_rate import estimar_frequencia_cardiaca
from .hrv import calcular_hrv
from .signal_metrics import calcular_metricas_sinal
from .respiratory_rate import estimar_taxa_respiratoria

__all__ = [
    "estimar_frequencia_cardiaca",
    "calcular_hrv",
    "calcular_metricas_sinal",
    "estimar_taxa_respiratoria",
]
