# PiEdge EduKit

![CI](https://github.com/olablom/PiEdge_EduKit/actions/workflows/ci.yml/badge.svg)

Ett reproducerbart undervisningskit för edge-ML på Raspberry Pi: träna liten bildklassificerare på PC → exportera till ONNX → benchmarka latens på Pi → styra GPIO (LED) med hysteresis.

## Snabbstart för lektion (30 min)

**För Python 3.12:** Öppna `index.html` i webbläsaren för komplett guide, eller kör:

```bash
# Skapa och aktivera virtuell miljö
bash scripts/setup_venv.sh
source .venv/bin/activate  # Linux/macOS
# eller .venv\Scripts\Activate.ps1  # Windows

# Kör hela lektionen
bash run_lesson.sh

# Verifiera resultat
python verify.py
```

Se `progress/receipt.json` för automatisk verifiering och kvitto.

## Dataset Policy

**Viktigt:** Detta projekt inkluderar INGA bilder eller binära filer. Användare måste tillhandahålla sina egna bilder.

### Datakrav:

- **Format:** JPG/PNG-bilder
- **Storlek:** Bilder kommer automatiskt att förbehandlas till 64x64 pixlar
- **Struktur:** Organisera bilder i klassmappar (`data/cat/`, `data/dog/`, `data/bird/`)
- **Mängd:** 60-120 bilder totalt (20-40 per klass)
- **Licens:** Använd CC0 eller egna bilder för utbildningssyfte

### Exempel på datastruktur:

```
data/
├── cat/
│   ├── cat1.jpg
│   ├── cat2.png
│   └── ...
├── dog/
│   ├── dog1.jpg
│   └── ...
└── bird/
    ├── bird1.jpg
    └── ...
```

## Snabbstart

### PC (Träning och utveckling)

**Alternativ A: Utan bilder (snabbstart med FakeData)**

```bash
# Klona och installera
git clone <repo-url>
cd piedge_edukit
pip install -e .

# Kör alla labbar med FakeData
make pc_all FAKEDATA=1

# Visa resultat i dashboard (valfritt)
streamlit run app.py
```

**Alternativ B: Med syntetiska bilder (transparens)**

```bash
# Klona och installera
git clone <repo-url>
cd piedge_edukit
pip install -e .

# Skapa syntetiskt dataset
python scripts/make_synthetic_dataset.py --root data --train-per-class 60 --val-per-class 20

# Kör alla labbar
make pc_all

# Visa resultat i dashboard (valfritt)
streamlit run app.py
```

### Raspberry Pi (64-bit OS krävs)

**Alternativ A: Med FakeData (ingen datafil behövs)**

```bash
# Installera Pi-specifika dependencies först (viktig ordning!)
sudo bash pi_setup/install_pi_requirements.sh

# Kör Lab 2: Latensbenchmark med FakeData
python -m piedge_edukit.benchmark --fakedata --model-path ./models/model.onnx --warmup 50 --runs 200

# Kör Lab 3: GPIO-styrning med FakeData (kräver GPIO-behörighet)
sudo usermod -aG gpio $USER
# Logga ut och in igen, sedan:
python -m piedge_edukit.gpio_control --fakedata --model-path ./models/model.onnx --target "class1" --duration 10 --no-simulate
```

**Alternativ B: Med syntetiska bilder**

```bash
# Installera Pi-specifika dependencies först (viktig ordning!)
sudo bash pi_setup/install_pi_requirements.sh

# Kör Lab 2: Latensbenchmark
python -m piedge_edukit.benchmark --model-path ./models/model.onnx --data-path ./data --warmup 50 --runs 200

# Kör Lab 3: GPIO-styrning (kräver GPIO-behörighet)
sudo usermod -aG gpio $USER
# Logga ut och in igen, sedan:
python -m piedge_edukit.gpio_control --model-path ./models/model.onnx --data-path ./data --target "class1" --duration 10 --no-simulate
```

## Laböversikt

- **Lab 1**: Träning → ONNX-export → ORT-verifiering
- **Lab 2**: Latensbenchmark (PC också tillåtet för metodik)
- **Lab 2b**: Bonus - Statisk INT8-kvantisering + jämförelse
- **Lab 3**: GPIO-styrning med hysteresis

## Krav

- **PC**: Python ≥3.10, PyTorch, ONNX Runtime
- **Pi**: 64-bit OS (aarch64), Python ≥3.10, ONNX Runtime CPU
- **Hårdvara**: Raspberry Pi med GPIO-pin 17 för LED

## Troubleshooting

### ONNX Runtime/NumPy-krock på Pi

```bash
# Installera onnxruntime först, sedan PyTorch
pip install onnxruntime
pip install torch torchvision
```

### GPIO-behörighet

```bash
sudo usermod -aG gpio $USER
# Logga ut och in igen
```

### MockGPIO för PC-testning

```bash
# Med FakeData (ingen datafil behövs)
python -m piedge_edukit.gpio_control --fakedata --simulate --model-path ./models/model.onnx --target "class1" --duration 5

# Med riktiga bilder
python -m piedge_edukit.gpio_control --simulate --model-path ./models/model.onnx --data-path ./data --target "class1" --duration 5
```

## CLI-kommandon

**Ingen PYTHONPATH krävs:** Installera paketet med `pip install -e .`, kör kommandona ovan. Om `piedge-*` saknas i PATH, använd `python -m piedge_edukit.<cmd>`.

### Träning

```bash
# Med riktiga bilder
python -m piedge_edukit.train --data-path <data> --output-dir <models> [--epochs N] [--batch-size N]

# Med FakeData (ingen datafil behövs)
python -m piedge_edukit.train --fakedata --output-dir <models> [--epochs N] [--batch-size N]
```

### Benchmark

```bash
# Med riktiga bilder
python -m piedge_edukit.benchmark --model-path <model.onnx> --data-path <data> [--warmup N] [--runs N]

# Med FakeData (ingen datafil behövs)
python -m piedge_edukit.benchmark --fakedata --model-path <model.onnx> [--warmup N] [--runs N]
```

### GPIO-styrning

```bash
# Med riktiga bilder
python -m piedge_edukit.gpio_control --model-path <model.onnx> --data-path <data> --target <klass> [--simulate] [--no-simulate] [--duration N]

# Med FakeData (ingen datafil behövs)
python -m piedge_edukit.gpio_control --fakedata --model-path <model.onnx> --target <klass> [--simulate] [--no-simulate] [--duration N]
```

**Viktiga flaggor:**

- `--fakedata`: Använd FakeData istället för riktiga bilder (ingen datafil behövs)
- `--simulate` (default): Använd MockGPIO för PC-testning
- `--no-simulate`: Använd riktig GPIO på Pi
- `--target`: Målklass för GPIO-styrning (obligatorisk)
- `--duration`: Körningstid i sekunder
- `--warmup`: Antal warmup-körningar (default: 50)
- `--runs`: Antal benchmark-körningar (default: 200)

## Makefile för enkelt körflöde

Använd `make` för att köra labbar enkelt:

```bash
# Visa tillgängliga kommandon
make help

# Kör alla labbar med FakeData
make pc_all FAKEDATA=1

# Kör alla labbar med syntetiska bilder
make synthetic
make pc_all

# Kör enskilda labbar
make lab1          # Träning och ONNX-export
make lab2          # Latensbenchmark
make lab2b         # Kvantisering (bonus)
make lab3          # GPIO-styrning (simulation)

# Rensa rapporter
make clean
```

## Streamlit Dashboard

För snygg visning av resultat:

```bash
# Starta dashboard
streamlit run app.py
```

Dashboard visar:

- **Overview**: Modellfiler och status
- **Training**: Träningskurvor (loss/accuracy)
- **Evaluation**: Confusion matrix och utvärdering
- **Benchmark**: Latensresultat och distributioner
- **Quantization**: Kvantiseringsjämförelse

## Projektstruktur

```
piedge_edukit/
├── models/                 # Tränade modeller och konfiguration
├── reports/               # Benchmark och GPIO-rapporter
├── labs/                  # Jupyter notebooks för labbar
├── pi_setup/             # Pi-specifika installationsskript
├── piedge_edukit/        # Huvudkod
└── data/                 # Träningsdata (ej inkluderad)
```

## Smoke Test

Kör automatisk smoke test för att verifiera installation:

```bash
# PC smoke test (fullständig)
python smoke_test.py

# Pi smoke test (minimal)
python smoke_test.py pi
```

## CLI-kommandon

### Träning

```bash
python -m piedge_edukit.train --data-path <data> --output-dir <models> [--epochs N] [--batch-size N]
```

### Benchmark

```bash
python -m piedge_edukit.benchmark --model-path <model.onnx> --data-path <data> [--warmup N] [--runs N]
```

### GPIO-styrning

```bash
python -m piedge_edukit.gpio_control --model-path <model.onnx> --data-path <data> --target <klass> [--simulate] [--no-simulate] [--duration N]
```

**Viktiga flaggor:**

- `--simulate` (default): Använd MockGPIO för PC-testning
- `--no-simulate`: Använd riktig GPIO på Pi
- `--target`: Målklass för GPIO-styrning (obligatorisk)
- `--duration`: Körningstid i sekunder
- `--warmup`: Antal warmup-körningar (default: 50)
- `--runs`: Antal benchmark-körningar (default: 200)

## Artefakter

Projektet genererar följande artefakter:

- `models/model.onnx` - Tränad ONNX-modell
- `models/labels.json` - Klassmappning
- `models/preprocess_config.json` - Preprocessing-konfiguration
- `reports/latency.csv` - Detaljerade benchmark-resultat
- `reports/latency_summary.txt` - Sammanfattning av latens
- `reports/latency_plot.png` - Benchmark-visualisering
- `reports/quantization_comparison.csv` - Kvantiseringsjämförelse (om Lab 2b körs)
- `reports/quantization_comparison.png` - Kvantiseringsvisualisering
- `reports/gpio_session.txt` - GPIO-sessionslogg
- `reports/gpio_history.png` - GPIO-historik

## Installation på Raspberry Pi

**Viktig installationsordning för att undvika ABI-konflikter:**

1. **ONNX Runtime först** (kritiskt för aarch64-kompatibilitet)
2. **NumPy** (pinned version för kompatibilitet)
3. **Övriga dependencies**

Installationsskriptet `pi_setup/install_pi_requirements.sh` följer denna ordning automatiskt.

## Troubleshooting

### ONNX Runtime/NumPy-krock på Pi

```bash
# Installera onnxruntime först, sedan PyTorch
pip install onnxruntime
pip install torch torchvision
```

### GPIO-behörighet

```bash
sudo usermod -aG gpio $USER
# Logga ut och in igen
```

### MockGPIO för PC-testning

```bash
python -m piedge_edukit.gpio_control --simulate --model-path ./models/model.onnx --target "bird" --duration 5
```

### Preprocessing hash mismatch

```bash
# Om preprocessing-validering misslyckas
rm models/preprocess_config.json
python -m piedge_edukit.train --data-path ./data --output-dir ./models
```

## Reproducerbarhet

För fullständig reproducerbarhet, se miljöfiler:

- `reports/env_pc.txt` - PC-miljö (genereras med `pip freeze`)
- `reports/env_pi.txt` - Pi-miljö (genereras på Pi efter installation)

Dessa filer skapas automatiskt vid smoke test-körning.
