# rPPG - Residência PIpos

O projeto é um único pacote Python localizado nesta pasta. A captura e a
análise são independentes: a captura salva um MP4 e a análise recebe o caminho
desse arquivo.

Execute a partir desta pasta:

```powershell
python main.py
python main.py --video "captures\teste01.mp4"
python main.py --capture-only
```

Também é suportada a execução a partir da pasta pai:

```powershell
python -m rPPG.main
```
