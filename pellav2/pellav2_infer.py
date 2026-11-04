"""Pellav2 audio deepfake detector: minimal inference code.

    pip install torch torchaudio transformers soundfile numpy scipy pyloudnorm
    python pellav2_infer.py clip1.wav clip2.wav ...

The weights are downloaded from the Hugging Face Hub on first use. To run from a
local copy instead, pass it first: python pellav2_infer.py pellav2_detector.pt clip1.wav ...
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

SR = 16000
CROP = 4 * SR
TARGET_LUFS = -23.0
HUB_REPO = "Sadanie/pellav2-audio-deepfake-detector"


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        local_config = Path(__file__).resolve().parent / "w2v-bert-2.0-config.json"
        self.backbone = (Wav2Vec2BertModel(Wav2Vec2BertConfig.from_json_file(str(local_config)))
                         if local_config.exists()
                         else Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0"))
        self.layer_weights = nn.Parameter(
            torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    @staticmethod
    def features(wav):
        import torchaudio.compliance.kaldi as K

        feats, lens = [], []
        with torch.autocast(device_type=wav.device.type, enabled=False):
            for x in wav.float():
                nz = torch.nonzero(x).flatten()
                n_s = int(nz[-1]) + 1 if len(nz) else len(x)
                f = K.fbank(x[None, :max(n_s, 400)] * 32768.0, num_mel_bins=80,
                            sample_frequency=SR, dither=0.0, window_type="povey")
                f = (f - f.mean(0, keepdim=True)) / torch.sqrt(f.var(0, keepdim=True) + 1e-7)
                n = f.shape[0] // 2 * 2
                feats.append(f[:n].reshape(n // 2, 160))
                lens.append(n // 2)
        F = max(lens)
        out = torch.zeros(len(feats), F, 160, device=wav.device)
        mask = torch.zeros(len(feats), F, dtype=torch.long, device=wav.device)
        for i, (f, n) in enumerate(zip(feats, lens)):
            out[i, :n] = f
            mask[i, :n] = 1
        return out, mask

    def forward(self, wav):
        feats, mask = self.features(wav)
        hs = self.backbone(input_features=feats, attention_mask=mask,
                           output_hidden_states=True).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        h = torch.stack(hs).mul(w[:, None, None, None]).sum(0)
        m = mask[:, :, None].float()
        h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(h).squeeze(-1)


def loudness(x):
    try:
        import pyloudnorm as pyln

        if len(x) >= int(0.4 * SR):
            v = pyln.Meter(SR).integrated_loudness(x.astype(np.float64))
            if np.isfinite(v):
                return float(v)
    except Exception:
        pass
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) - 0.691)


def load_clip(path):
    import torchaudio
    from scipy.signal import butter, sosfiltfilt

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    wav = wav - wav.mean()
    if len(wav) > 3 * SR // 100:
        wav = sosfiltfilt(butter(2, 20.0, btype="highpass", fs=SR, output="sos"), wav)
    wav = wav.astype(np.float32)
    if len(wav) > CROP:
        off = (len(wav) - CROP) // 2
        wav = wav[off: off + CROP]
    if len(wav):
        gain = float(np.clip(TARGET_LUFS - loudness(wav), -40.0, 40.0))
        wav = np.clip(wav * 10 ** (gain / 20), -1.0, 1.0).astype(np.float32)
    if len(wav) < CROP:
        wav = np.concatenate([wav, np.zeros(CROP - len(wav), np.float32)])
    return wav


def hub_weights():
    from huggingface_hub import hf_hub_download

    hf_hub_download(HUB_REPO, "config.json")
    return hf_hub_download(HUB_REPO, "pellav2_detector.pt")


def main():
    args = sys.argv[1:]
    if args and args[0].endswith(".pt"):
        weights, files = args[0], args[1:]
    else:
        weights, files = hub_weights(), args
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Detector().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    with torch.no_grad():
        for f in files:
            x = torch.from_numpy(load_clip(f))[None].to(device)
            p = torch.sigmoid(model(x)).item()
            print(f"{f}: p_fake={p:.3f} -> {'FAKE' if p >= 0.5 else 'real'}")


if __name__ == "__main__":
    main()
