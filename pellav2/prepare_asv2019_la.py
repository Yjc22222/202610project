"""Build a labeled manifest from the local ASVspoof 2019 LA evaluation protocol."""
import csv
from collections import Counter
from pathlib import Path

root = Path(r"D:\Microsoft VS Code\202610project\aasist-main\LA")
protocol = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
audio = root / "ASVspoof2019_LA_eval" / "flac"
output = Path(__file__).with_name("asv2019_la_eval_manifest.csv")
rows = []
missing = []
for line in protocol.read_text(encoding="utf-8").splitlines():
    speaker, clip_id, _, attack, label = line.split()
    path = audio / f"{clip_id}.flac"
    if not path.is_file():
        missing.append(str(path))
    rows.append(dict(path=str(path), label=label, group=speaker,
                     type=attack if attack != "-" else "Bonafide"))
if missing:
    raise FileNotFoundError(f"{len(missing)} labeled audio files missing; first: {missing[0]}")
with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "type"])
    writer.writeheader()
    writer.writerows(rows)
print(f"manifest={output} count={len(rows)} labels={dict(Counter(r['label'] for r in rows))}")
