# Lab-guider för PiEdge EduKit

## Lab 1: Träning och ONNX-export

**Mål**: Träna MobileNetV2-modell och exportera till ONNX

**Kommando**:

```bash
python -m piedge_edukit.train --data-path ./data --output-dir ./models
```

**Förväntade utfall**:

- `models/model.onnx` - Tränad ONNX-modell
- `models/labels.json` - Klassmappning
- `models/preprocess_config.json` - Preprocessing-konfiguration

## Lab 2: Latensbenchmark

**Mål**: Benchmarka modellens latens på PC och Pi

**Kommando**:

```bash
python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data
```

**Förväntade utfall**:

- `reports/latency.csv` - Detaljerade resultat
- `reports/latency_summary.txt` - Sammanfattning
- `reports/latency_plot.png` - Visualisering

## Lab 2b: Statisk INT8-kvantisering (Bonus)

**Mål**: Kvantisera modell och jämför prestanda

**Kommando**:

```bash
python -m piedge_edukit.quantization --model-path ./models/model.onnx --data-path ./data
```

**Förväntade utfall**:

- `models/model_static.onnx` - Kvantiserad modell
- `reports/quantization_comparison.csv` - Jämförelsetabell
- `reports/quantization_comparison.png` - Visualisering

## Lab 3: GPIO-styrning med hysteresis

**Mål**: Integrera ML med GPIO-styrning

**Kommando (PC med MockGPIO)**:

```bash
python -m piedge_edukit.gpio_control --model-path ./models/model.onnx --data-path ./data --simulate
```

**Kommando (Pi med riktig GPIO)**:

```bash
python -m piedge_edukit.gpio_control --model-path ./models/model.onnx --data-path ./data --gpio-pin 17
```

**Förväntade utfall**:

- `reports/gpio_session.txt` - Sessionslogg
- `reports/gpio_history.png` - GPIO-historik
- Realtidsstyrning av LED baserat på ML-confidence

## Allmänna tips

- Kör alltid Lab 1 först för att skapa modellen
- Använd `--simulate` för GPIO-testning på PC
- Kontrollera att alla artefakter skapas korrekt
- Läs rapportfilerna för detaljerade resultat
