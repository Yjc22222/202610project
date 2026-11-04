import argparse, csv, gc, json, random, time
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from pellav2_infer import Detector, load_clip
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
from torch import nn

def sample(rows, label, n, rng):
    groups = defaultdict(list)
    for r in rows:
        if r["label"] == label:
            groups[r.get("type", "")].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out = []
    while groups and len(out) < n:
        for k in sorted(tuple(groups)):
            out.append(groups[k].pop())
            if not groups[k]:
                del groups[k]
            if len(out) == n:
                break
    return out

def metrics(rows):
    ok = [r for r in rows if not r["error"]]
    real = [r for r in ok if r["label"] == "bonafide"]
    fake = [r for r in ok if r["label"] == "spoof"]
    tp = sum(r["predicted"] == "spoof" for r in fake)
    tn = sum(r["predicted"] == "bonafide" for r in real)
    tpr = tp / len(fake) if fake else None
    tnr = tn / len(real) if real else None
    return dict(total=len(rows), scored=len(ok), failed=len(rows)-len(ok),
                accuracy=(tp+tn)/len(ok) if ok else None,
                balanced_accuracy=(tpr+tnr)/2 if fake and real else None,
                fake_recall=tpr, real_false_positive_rate=1-tnr if real else None,
                TP_fake=tp, TN_real=tn, FP_real_as_fake=len(real)-tn,
                FN_fake_as_real=len(fake)-tp)

p = argparse.ArgumentParser()
p.add_argument("--weights", type=Path, required=True)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--per-class", type=int, default=1000)
p.add_argument("--seed", type=int, default=20261102)
a = p.parse_args()
with a.manifest.open(encoding="utf-8-sig", newline="") as f:
    all_rows = list(csv.DictReader(f))
rng = random.Random(a.seed)
rows = sample(all_rows, "bonafide", a.per_class, rng)
rows += sample(all_rows, "spoof", a.per_class, rng)
rng.shuffle(rows)
if not rows:
    raise RuntimeError("No audio selected")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {len(rows)} / {len(all_rows)}; device={device}", flush=True)
start = time.time()
local_config = Path(__file__).resolve().parent / "w2v-bert-2.0-config.json"
config = (Wav2Vec2BertConfig.from_json_file(str(local_config))
          if local_config.exists()
          else Wav2Vec2BertConfig.from_pretrained("facebook/w2v-bert-2.0"))
class CheckpointDetector(Detector):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = Wav2Vec2BertModel(config)
        self.layer_weights = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        self.head = nn.Linear(config.hidden_size, 1)
model = CheckpointDetector()
state = torch.load(a.weights, map_location="cpu", weights_only=True)
model.load_state_dict(state)
del state
gc.collect()
model = model.to(device).eval()
print(f"Model loaded in {time.time()-start:.1f}s", flush=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
out = []
with a.output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path","label","group","type","p_fake","predicted","error"])
    writer.writeheader()
    for i,r in enumerate(rows,1):
        x = {k:r.get(k,"") for k in ("path","label","group","type")}
        try:
            waveform = torch.from_numpy(load_clip(r["path"]))[None].to(device)
            with torch.inference_mode():
                score = torch.sigmoid(model(waveform)).item()
            x.update(p_fake=score, predicted="spoof" if score >= 0.5 else "bonafide", error="")
        except Exception as e:
            x.update(p_fake="", predicted="", error=f"{type(e).__name__}: {e}")
        writer.writerow(x)
        out.append(x)
        if i % 100 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} ({i/len(rows):.1%})", flush=True)
summary = dict(model="Sadanie/pellav2-audio-deepfake-detector",
               threshold=0.5, threshold_source="published code, no test tuning",
               preprocessing="published load_clip, 16 kHz mono, centered 4 s",
               manifest=str(a.manifest.resolve()), seed=a.seed,
               label_counts=dict(Counter(r["label"] for r in rows)),
               type_counts=dict(Counter(r["type"] for r in rows)),
               metrics=metrics(out),
               by_type={k:metrics([r for r in out if r["type"]==k]) for k in sorted({r["type"] for r in out})},
               errors=[r["error"] for r in out if r["error"]][:10],
               seconds=time.time()-start)
a.output.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"metrics":summary["metrics"],"seconds":summary["seconds"]}),flush=True)


