# Changelog

Alla betydande ändringar i detta projekt kommer att dokumenteras i denna fil.

Formatet är baserat på [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
och detta projekt följer [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-30

### Added

- Komplett PiEdge EduKit för edge-ML på Raspberry Pi
- Träning av MobileNetV2-bildklassificerare med PyTorch
- ONNX-export med opset ≥17 och dynamisk batch-storlek
- Latensbenchmark med ONNX Runtime (p50/p95/mean/std)
- Static INT8-kvantisering med graceful fallback
- GPIO-styrning med hysteresis och debouncing
- MockGPIO för PC-testning utan hårdvara
- Preprocessing-hash validering för konsistens
- Labels-integritet validering
- Komplett CLI med både console_scripts och python -m
- Smoke test för automatisk verifiering
- Detaljerad dokumentation och troubleshooting

### Fixed

- Unicode-kompatibilitet för Windows-konsoler
- CUDA/CPU-konflikter vid ONNX-export
- JSON serialisering av heltalsnycklar i labels
- Quantization fallback-hantering
- Artefaktkonsistens (alla rapporter i reports/)

### Documentation

- Komplett README med snabbstart för PC och Pi
- CLI-flaggor dokumentation
- Installationsordning för Pi (ORT → numpy → resten)
- Troubleshooting-sektion med vanliga problem
- Reproducerbarhet med miljöfiler

### Technical

- Python 3.12 only requirement
- Pinned versions i requirements-pc.txt och requirements-pi.txt
- Versioneringsstämplar i alla rapporter
- Determinism med seeds och cudnn.deterministic=True
- Komplett artefaktkontrakt (models/ och reports/)
