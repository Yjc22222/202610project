"""Decode application audio without changing the source file."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.aif', '.aiff'}

class AudioError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def load_audio(path, settings):
    path = Path(path)
    if not path.is_file():
        raise AudioError('file_not_found', str(path))
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError('unsupported_format', 'Unsupported audio extension: ' + path.suffix)
    if path.stat().st_size > settings['max_file_mb'] * 1024**2:
        raise AudioError('file_too_large', 'File exceeds configured size limit.')
    max_seconds = settings['max_duration_seconds']
    backend = 'soundfile'
    try:
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
            raise AudioError('empty_audio', 'No audio samples.')
        if info.duration > max_seconds:
            raise AudioError('audio_too_long', 'Audio exceeds configured duration limit.')
        audio, sample_rate = sf.read(str(path), dtype='float32', always_2d=True)
        channels = audio.shape[1]
        audio = audio.mean(axis=1, dtype=np.float32)
    except AudioError:
        raise
    except (RuntimeError, OSError) as decode_error:
        ffmpeg = settings.get('ffmpeg') or shutil.which('ffmpeg')
        if not ffmpeg:
            raise AudioError('decode_failed', 'Cannot decode audio with soundfile. For M4A/AAC, install FFmpeg or set audio.ffmpeg. ' + str(decode_error)) from decode_error
        sample_rate = settings['sample_rate']
        # Decode at most the duration limit plus one second, then reject overflow.
        command = [str(ffmpeg), '-nostdin', '-v', 'error', '-i', str(path), '-vn',
                   '-t', str(max_seconds + 1), '-ac', '1', '-ar', str(sample_rate),
                   '-f', 'f32le', 'pipe:1']
        try:
            completed = subprocess.run(command, capture_output=True, timeout=settings['decode_timeout_seconds'], check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AudioError('decode_failed', str(error)) from error
        if completed.returncode:
            raise AudioError('decode_failed', completed.stderr.decode('utf-8', errors='replace')[-1000:])
        audio = np.frombuffer(completed.stdout, dtype='<f4').copy()
        channels = None
        backend = 'ffmpeg'
    if audio.size == 0:
        raise AudioError('empty_audio', 'No decoded audio samples.')
    if not np.isfinite(audio).all():
        raise AudioError('invalid_samples', 'Audio contains NaN or infinite samples.')
    duration = len(audio) / sample_rate
    if duration > max_seconds:
        raise AudioError('audio_too_long', 'Audio exceeds configured duration limit.')
    if duration < settings['min_duration_seconds']:
        raise AudioError('audio_too_short', 'Audio is too short for configured detection policy.')
    if float(np.max(np.abs(audio))) <= 1e-8:
        raise AudioError('silent_audio', 'Audio contains no usable signal.')
    target_rate = settings['sample_rate']
    original_rate = sample_rate
    if sample_rate != target_rate:
        audio = resample(torch.from_numpy(audio), sample_rate, target_rate).numpy()
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    return audio, {'duration_seconds': duration, 'decoded_sample_rate': original_rate,
                   'decoded_channels': channels, 'sample_rate': target_rate,
                   'decoder': backend, 'samples': len(audio)}


def decoder_identity(settings):
    """Record decoder versions so calibration cannot silently change backends."""
    import hashlib
    identity = {'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__, 'ffmpeg': None}
    executable = settings.get('ffmpeg') or shutil.which('ffmpeg')
    if executable:
        try:
            result = subprocess.run([str(executable), '-version'], capture_output=True, timeout=5, check=True)
        except (OSError, subprocess.SubprocessError) as error:
            raise AudioError('ffmpeg_unavailable', str(error)) from error
        identity['ffmpeg'] = hashlib.sha256(result.stdout).hexdigest()
    return identity
