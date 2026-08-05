# rPPG — Residência Pipos

Sistema de fotopletismografia remota (rPPG): a partir de um vídeo do rosto
(captura via webcam ou arquivo já gravado), estima a **frequência cardíaca**, a
**variabilidade da frequência cardíaca (HRV)** e métricas de qualidade do
sinal. 

---

## Como funciona

1. **Detecção facial** — MediaPipe Face Landmarker localiza os pontos do rosto em cada frame (bochechas e testa)
2. **Extração das ROIs** — três regiões de interesse são recortadas e a média espacial de R, G, B é calculada dentro de cada uma.
3. **Pré-processamento** — suavização para remover ruído de câmera/movimento.
4. **Extração do sinal rPPG** — três algoritmos rodam em paralelo sobre cada ROI:
   - **CHROM** (De Haan & Jeanne, 2013)
   - **POS** (Wang et al., 2017) 
   - **GREEN** (Verkruysse et al., 2008)
5. **Combinação** — os três sinais são padronizados (z-score), alinhados e combinados por pesos configuráveis, primeiro entre algoritmos e depois entre as ROIs.
6. **Filtragem final** — os sinais passam por um filtro Butterworth passa-banda que restringe o sinal combinado à faixa fisiológica de frequência cardíaca (42–240 bpm).
7. **Biomarcadores** — a frequência cardíaca (pico espectral via FFT), HRV (SDNN, RMSSD, pNN50 a partir dos intervalos entre picos) e métricas de qualidade do sinal (SNR, concentração espectral, amplitude, energia) são calculados.
8. **Relatório** — resultado final dos biomarcadores impresso no console, contendo benchmarks comparando as métricas do CHROM/POS/GREEN, e opcionalmente, gráficos de comparação entre eles.

---

## Pré-requisitos

- Python 3.8+
- pip
- webcam (para captura) 

---

## Instalação

```bash
git clone https://github.com/AdaoRios/rPPG---Estatisca.git
cd rPPG
pip install -r requirements.txt
```

O modelo de landmarks faciais do MediaPipe já está incluído em
`assets/face_landmarker.task`.

---

## Como usar

Execute a partir desta pasta (`rPPG/`):

```bash
# Menu interativo (captura, análise, ou ambos)
python main.py

# Analisar um vídeo já existente
python main.py --video "data/captures/video_2026-01-01_10-00-00.mp4"

# Só capturar (sem analisar)
python main.py --capture-only

# Capturar e analisar em sequência, com duração e câmera customizadas
python main.py --duration 20 --camera 1
```

### Menu interativo

Ao rodar `python main.py`, um menu é exibido:

```
1 - Capturar novo vídeo
2 - Analisar vídeo existente
3 - Capturar e analisar
4 - Benchmark (em breve)
5 - Configurações (em breve)
0 - Sair
```

### Argumentos de linha de comando

| Argumento | Descrição | Padrão |
|---|---|---|
| `--video <caminho>` | Analisa um vídeo já existente em vez de capturar | — |
| `--duration <s>` | Duração da captura em segundos | `30.0` |
| `--camera <índice>` | Índice da câmera a ser usada | `0` |
| `--capture-only` | Apenas grava o vídeo, sem rodar a análise | `False` |

Vídeos capturados são salvos em `data/captures/`, com o nome `video_<data>_<hora>.mp4` (a pasta é criada automaticamente na primeira captura).

---

## Configuração

Todos os parâmetros do pipeline ficam centralizados em `config.py`:

```python
MODEL_PATH = "assets/face_landmarker.task"
DEBUG_COMPARE_ALGORITHMS = True   # liga/desliga os gráficos de diagnóstico

HR_LOW_HZ = 0.7    # 42 bpm — limite inferior aceito para a FC
HR_HIGH_HZ = 4.0   # 240 bpm — limite superior aceito para a FC

METHOD_WEIGHTS = {"chrom": 0.4, "pos": 0.4, "green": 0.2}

ROI_WEIGHTS = {
    "testa": 0.4,
    "bochecha_esquerda": 0.3,
    "bochecha_direita": 0.3,
}

ROI_POINTS = { ... }  # índices dos landmarks do MediaPipe para cada ROI
```

---

## Arquitetura do projeto

```
rPPG/
├── main.py                        # CLI e menu interativo
├── config.py                      # constantes e pesos do pipeline
├── requirements.txt
├── README.md
├── assets/
│   └── face_landmarker.task       # modelo de landmarks faciais (MediaPipe)
│
├── capture/                       # pipeline de captura (independente da análise)
│   ├── captura.py                 #   webcam, qualidade do quadro e gravação MP4
│   └── __init__.py                #   API pública de captura
│
├── analysis/                      # pipeline de análise (independente da captura)
│   └── analyze_video.py           #   lê um .mp4 e orquestra o pipeline completo de rPPG usando extractors/combination.py
│
├── roi/
│   ├── facial.py                  #   landmarks, máscaras e médias RGB das ROIs
│   └── __init__.py                #   API pública de regiões faciais
│
├── preprocessing/
│   ├── processamento.py           #   suavização e filtro passa-banda
│   └── __init__.py                #   API pública de pré-processamento
│
├── extractors/                     # extração e fusão do sinal rPPG
│   ├── algorithms.py               #   CHROM, POS e GREEN
│   ├── combination.py              #   fusão ponderada e benchmarks
│   └── __init__.py                 #   API pública dos extratores
│
├── biomarkers/
│   ├── heart_rate.py               #   frequência cardíaca via pico da FFT
│   ├── hrv.py                      #   SDNN, RMSSD, pNN50
│   ├── signal_metrics.py           #   métricas de qualidade de sinal e FFT
│   └── respiratory_rate.py         #   ainda não implementado
│
├── reports/
│   ├── apresentacao.py             #   implementação de console e gráficos
│   └── __init__.py                 #   API pública de apresentação
│
└── utils/
    └── models.py                   #   dataclass AnalysisResult
```

---

## Saída da análise

`analyze_video()` retorna um `AnalysisResult` com:

- `heart_rate` — frequência cardíaca estimada (bpm)
- `hrv` — dicionário com `SDNN_ms`, `RMSSD_ms`, `pNN50_%` e `n_batimentos_detectados`
- `respiratory_rate` — não implementado
- `signal_metrics` — `snr`, `spectral_concentration`, `fft_peak`, `amplitude`, `std`, `energy`
- `fps`, `duration`, `valid_frames` — informações sobre a captura

O relatório é impresso automaticamente via `print_report()` ao rodar `main.py`. Quando `DEBUG_COMPARE_ALGORITHMS = True` (ou `debug=True` em `combine_roi_and_methods`), também são impressos benchmarks comparando CHROM, POS e GREEN por ROI e por algoritmo, além de um gráfico consolidado.


## Referências

- de Haan, G., & Jeanne, V. (2013). *Robust pulse rate from chrominance-based rPPG*. IEEE Transactions on Biomedical Engineering.
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). *Algorithmic principles of remote PPG*. IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). *Remote plethysmographic imaging using ambient light*. Optics Express.
