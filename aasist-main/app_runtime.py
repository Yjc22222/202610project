"""Frozen AASIST inference; no training or ASVspoof protocols are required."""
import hashlib
import inspect
import json
import time
from pathlib import Path
import numpy as np
import torch
import torchaudio
from audio_io import AudioError, load_audio, decoder_identity
from segmenter import make_segments, aggregate_scores
from app_utils import file_sha256, read_json
from models.AASIST import Model


class AASISTDetector:
    def __init__(self, config_path, require_profile=True, device=None):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        base = self.config_path.parent
        self.architecture_path = (base / self.config['architecture_config']).resolve()
        self.weights_path = (base / self.config['weights']).resolve()
        self.profile_path = (base / self.config['threshold_profile']).resolve()
        self.architecture = read_json(self.architecture_path)['model_config']
        audio = self.config['audio']
        segments = self.config['segmentation']
        if self.architecture['architecture'] != 'AASIST':
            raise ValueError('Only the AASIST architecture is supported.')
        if audio['sample_rate'] != 16000 or segments['window_samples'] != 64600:
            raise ValueError('This checkpoint requires 16 kHz audio and 64600-sample windows.')
        if not 0 < segments['hop_samples'] <= segments['window_samples']:
            raise ValueError('Invalid hop length.')
        if segments['mode'] not in ('first', 'sliding') or segments['aggregation'] not in ('mean', 'median', 'q10'):
            raise ValueError('Invalid segmentation or aggregation setting.')
        if not 0 < audio['min_duration_seconds'] < audio['max_duration_seconds']:
            raise ValueError('Invalid duration limits.')
        if audio['max_file_mb'] <= 0 or audio['decode_timeout_seconds'] <= 0:
            raise ValueError('Invalid audio resource limits.')
        if not 1 <= self.config['batch_size'] <= 128 or not 1 <= self.config['cpu_threads'] <= 64:
            raise ValueError('Invalid batch size or CPU thread count.')
        selected = device or self.config['device']
        if selected not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu, or cuda.')
        self.device = 'cuda' if selected == 'auto' and torch.cuda.is_available() else ('cpu' if selected == 'auto' else selected)
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu or auto.')
        torch.set_num_threads(self.config['cpu_threads'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model = Model(self.architecture).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.weights_sha256 = file_sha256(self.weights_path)
        self.identity = {
            'algorithm': 'aasist-application-v1', 'weights_sha256': self.weights_sha256,
            'architecture': self.architecture, 'audio': audio,
            'segmentation': segments, 'score': 'bonafide_logit',
            'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
            'decoder_versions': decoder_identity(audio),
            'source_sha256': {name: file_sha256(Path(__file__).parent / name)
                              for name in ('audio_io.py', 'segmenter.py', 'app_runtime.py')},
            'model_source_sha256': file_sha256(inspect.getfile(Model)),
        }
        canonical = json.dumps(self.identity, sort_keys=True, ensure_ascii=True).encode('utf-8')
        self.pipeline_id = hashlib.sha256(canonical).hexdigest()
        self.profile = None
        self.threshold = None
        if require_profile:
            if not self.profile_path.is_file():
                raise ValueError('Threshold profile is missing. Run calibrate.py first, or use --scores-only for unclassified scores.')
            self.profile = read_json(self.profile_path)
            if self.profile.get('pipeline_id') != self.pipeline_id:
                raise ValueError('Threshold profile does not match the weights/preprocessing/window configuration. Recalibrate.')
            self.threshold = float(self.profile['threshold'])
            if not np.isfinite(self.threshold):
                raise ValueError('Threshold must be finite.')

    def score_file(self, path, include_segments=True):
        started = time.monotonic()
        result = {'file': str(Path(path).resolve()), 'status': 'error', 'prediction': None,
                  'raw_score': None, 'threshold': self.threshold, 'pipeline_id': self.pipeline_id,
                  'weights_sha256': self.weights_sha256}
        try:
            result['sha256'] = file_sha256(path)
            audio, metadata = load_audio(path, self.config['audio'])
            result['audio'] = metadata
            settings = self.config['segmentation']
            windows = make_segments(audio, settings['window_samples'], settings['hop_samples'], settings['mode'])
            scores = []
            batch_size = self.config['batch_size']
            with torch.inference_mode():
                for start in range(0, len(windows), batch_size):
                    batch = np.stack([part.samples for part in windows[start:start + batch_size]])
                    _, logits = self.model(torch.from_numpy(batch).to(self.device))
                    scores.extend(logits[:, 1].cpu().numpy().astype(float).tolist())
            score = aggregate_scores(scores, settings['aggregation'])
            result.update(status='ok' if self.threshold is not None else 'scored', raw_score=score,
                          segment_count=len(windows))
            if self.threshold is not None:
                result['prediction'] = 'bonafide' if score > self.threshold else 'spoof'
            if include_segments:
                rate = self.config['audio']['sample_rate']
                result['segments'] = [dict(start_seconds=part.start_sample / rate,
                                           end_seconds=part.end_sample / rate, raw_score=value)
                                      for part, value in zip(windows, scores)]
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            result['error'] = {'code': getattr(error, 'code', type(error).__name__), 'message': str(error)}
        result['elapsed_seconds'] = time.monotonic() - started
        return result

    def score_manifest(self, rows):
        records = []
        started = time.monotonic()
        for index, row in enumerate(rows, 1):
            record = self.score_file(row['path'], include_segments=False)
            record.update({key: row[key] for key in ('label', 'group', 'type')})
            records.append(record)
            if index % 25 == 0 or index == len(rows):
                elapsed = time.monotonic() - started
                print('Processed {}/{} files ({:.1f} files/s)'.format(index, len(rows), index / max(elapsed, .001)), flush=True)
        return records
