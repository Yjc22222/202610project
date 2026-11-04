import copy
import os
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_io import AudioError, load_audio
from segmenter import make_segments, aggregate_scores
from app_metrics import classification_metrics, threshold_curve, select_threshold
from app_utils import read_manifest, write_json
from app_runtime import AASISTDetector

PROJECT = Path(os.environ.get('AASIST_TEST_PROJECT', str(ROOT)))
BASE = json.loads((ROOT/'config'/'inference.json').read_text(encoding='utf-8-sig'))

class FunctionalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_window_coverage_and_tail(self):
        audio = np.arange(200001,dtype=np.float32)
        windows = make_segments(audio)
        covered = np.zeros(len(audio),dtype=bool)
        for part in windows:
            self.assertEqual(len(part.samples),64600)
            covered[part.start_sample:part.end_sample]=True
        self.assertTrue(covered.all())
        self.assertEqual(windows[-1].end_sample,len(audio))
    def test_short_and_first_window_match_original(self):
        from data_utils import pad
        for n in (1,1000,64600,80001):
            audio=np.arange(n,dtype=np.float32)
            actual=make_segments(audio,mode='first')[0].samples
            np.testing.assert_array_equal(actual,pad(audio))
    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            make_segments(np.ones(1),hop_samples=70000)
        with self.assertRaises(ValueError):
            make_segments(np.array([]))
        with self.assertRaises(ValueError):
            aggregate_scores([float('nan')])
    def test_stereo_resampling_and_metadata(self):
        rate=48000
        mono=.1*np.sin(2*np.pi*440*np.arange(rate)/rate)
        audio=self.root/'双声道.wav'
        sf.write(audio,np.column_stack([mono,mono*.5]),rate,subtype='FLOAT')
        actual,metadata=load_audio(audio,BASE['audio'])
        self.assertEqual(actual.shape,(16000,))
        self.assertEqual(actual.dtype,np.float32)
        self.assertEqual(metadata['decoded_channels'],2)
        self.assertEqual(metadata['decoded_sample_rate'],48000)
        self.assertAlmostEqual(float(np.max(np.abs(actual))),.075,delta=.003)
    def test_invalid_silent_and_too_short_audio(self):
        for name,data,expected in [('silence.wav',np.zeros(16000),'silent_audio'),
                                   ('short.wav',np.ones(100),'audio_too_short')]:
            audio=self.root/name
            sf.write(audio,data,16000)
            with self.assertRaises(AudioError) as result:
                load_audio(audio,BASE['audio'])
            self.assertEqual(result.exception.code,expected)
        with self.assertRaises(AudioError) as result:
            load_audio(self.root/'missing.wav',BASE['audio'])
        self.assertEqual(result.exception.code,'file_not_found')
    def test_duration_limit_and_corrupt_audio(self):
        audio=self.root/'long.wav'
        sf.write(audio,np.ones(32000),16000)
        settings=copy.deepcopy(BASE['audio'])
        settings['max_duration_seconds']=1
        with self.assertRaises(AudioError) as result:
            load_audio(audio,settings)
        self.assertEqual(result.exception.code,'audio_too_long')
        corrupt=self.root/'corrupt.wav'
        corrupt.write_bytes(b'not a WAV')
        with self.assertRaises(AudioError):
            load_audio(corrupt,BASE['audio'])
    def test_metrics_failures_do_not_disappear(self):
        pairs=[('spoof','spoof'),('bonafide','bonafide'),('bonafide','spoof'),('spoof','bonafide'),('spoof',None),('bonafide',None)]
        records=[{'label':a,'prediction':b,'status':'ok' if b else 'error'} for a,b in pairs]
        metrics=classification_metrics(records)
        self.assertEqual(metrics['failed'],2)
        self.assertAlmostEqual(metrics['accuracy_on_all'],2/6)
        self.assertAlmostEqual(metrics['accuracy_on_scored'],2/4)
        self.assertAlmostEqual(metrics['spoof_recall_on_all'],1/3)
        self.assertAlmostEqual(metrics['spoof_precision'],.5)
        self.assertAlmostEqual(metrics['spoof_f1_on_all'],.4)
    def test_threshold_ties_and_direction(self):
        scores=[.2,.2,.5,.9,1.0]
        labels=['spoof','bonafide','spoof','bonafide','bonafide']
        thresholds,frr,far=threshold_curve(scores,labels)
        for threshold,reject,accept in zip(thresholds,frr,far):
            self.assertAlmostEqual(reject,sum(s<=threshold for s,l in zip(scores,labels) if l=='bonafide')/3)
            self.assertAlmostEqual(accept,sum(s>threshold for s,l in zip(scores,labels) if l=='spoof')/2)
        selected=select_threshold(scores,labels)
        self.assertAlmostEqual(selected['validation_balanced_accuracy'],max(1-(frr+far)/2))
    def test_manifest_labels_and_duplicate_rejection(self):
        manifest=self.root/'input.csv'
        manifest.write_text('path,label\na.wav,spoof\nb.wav,bonafide\n',encoding='utf-8')
        rows=read_manifest(manifest)
        self.assertEqual(rows[0]['path'],str((self.root/'a.wav').resolve()))
        manifest.write_text('path,label\na.wav,spoof\na.wav,bonafide\n',encoding='utf-8')
        with self.assertRaises(ValueError):
            read_manifest(manifest)

class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests')
        cls.root=Path(cls.temp.name)
        cls.config=copy.deepcopy(BASE)
        cls.config['architecture_config']=str(PROJECT/'config'/'AASIST.conf')
        cls.config['weights']=str(PROJECT/'models'/'weights'/'AASIST.pth')
        cls.config['threshold_profile']=str(cls.root/'threshold.json')
        cls.config['segmentation']['mode']='first'
        cls.path=cls.root/'inference.json'
        write_json(cls.path,cls.config)
        cls.detector=AASISTDetector(cls.path,require_profile=False)
    @classmethod
    def tearDownClass(cls):
        del cls.detector
        torch.cuda.empty_cache()
        cls.temp.cleanup()
    def test_real_audio_matches_original_preprocessing(self):
        from data_utils import pad
        protocol=PROJECT/'LA'/'ASVspoof2019_LA_cm_protocols'/'ASVspoof2019.LA.cm.eval.trl.txt'
        key=protocol.read_text().splitlines()[0].split()[1]
        audio=PROJECT/'LA'/'ASVspoof2019_LA_eval'/'flac'/(key+'.flac')
        waveform,_=sf.read(audio)
        tensor=torch.Tensor(pad(waveform)).unsqueeze(0).to(self.detector.device)
        with torch.inference_mode():
            expected=float(self.detector.model(tensor)[1][0,1].cpu())
        result=self.detector.score_file(audio)
        self.assertEqual(result['status'],'scored')
        self.assertIsNone(result['prediction'])
        self.assertAlmostEqual(result['raw_score'],expected,places=5)
    def test_missing_file_is_not_bonafide(self):
        result=self.detector.score_file(self.root/'missing.wav')
        self.assertEqual(result['status'],'error')
        self.assertIsNone(result['prediction'])
    def test_profile_matching_and_mismatch_rejection(self):
        write_json(self.root/'threshold.json',{'pipeline_id':self.detector.pipeline_id,'threshold':.123})
        matching=AASISTDetector(self.path)
        self.assertEqual(matching.threshold,.123)
        del matching
        changed=copy.deepcopy(self.config)
        changed['segmentation']['aggregation']='q10'
        changed_path=self.root/'changed.json'
        write_json(changed_path,changed)
        with self.assertRaisesRegex(ValueError,'does not match'):
            AASISTDetector(changed_path)

if __name__=='__main__':
    unittest.main(verbosity=2)
