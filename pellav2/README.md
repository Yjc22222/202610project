---
license: mit
language: en
tags:
  - audio
  - deepfake-detection
  - anti-spoofing
---

# Pellav2: audio deepfake detector

## Files

- `pellav2_detector.pt`: the model weights.
- `pellav2_infer.py`: ready to run inference script (model class included).
- `config.json`: model identifier.

## Usage

```bash
pip install torch torchaudio transformers soundfile numpy scipy pyloudnorm
python pellav2_infer.py your_clip.wav
```

The weights are downloaded from the Hub on first use.
