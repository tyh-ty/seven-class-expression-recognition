"""Lightweight CPU temporal inference with bounded, isolated camera-session history."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from threading import Lock
import time

import numpy as np
import onnxruntime as ort


def box_iou(a, b):
    width = max(0, min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x']))
    height = max(0, min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    intersection = width*height
    return intersection/max(1, a['width']*a['height']+b['width']*b['height']-intersection)


class TemporalRuntime:
    def __init__(self, graph_path: Path, visual_graph_path: Path, max_sessions=16):
        if max_sessions < 1:
            raise ValueError('Session bound must be positive')
        self.metadata = json.loads(graph_path.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(graph_path.read_bytes()).hexdigest() != self.metadata['sha256']:
            raise ValueError('Temporal graph checksum mismatch')
        if hashlib.sha256(visual_graph_path.read_bytes()).hexdigest() != self.metadata['base_model_sha256']:
            raise ValueError('Temporal and visual graph fingerprints do not match')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape != [1, 8, 7] or self.session.get_outputs()[0].shape != [1, 7]:
            raise ValueError('Unexpected temporal graph dimensions')
        self.contexts = OrderedDict()
        self.max_sessions = max_sessions
        self.lock = Lock()

    def reset(self, session_id):
        with self.lock:
            self.contexts.pop(session_id, None)

    def transform(self, session_id, probabilities, bbox, now=None):
        """No session means a still-image request; return its original scores unchanged."""
        if not session_id:
            return probabilities
        if not isinstance(session_id, str) or len(session_id) > 80:
            raise ValueError('Invalid camera session identifier')
        now = time.perf_counter() if now is None else float(now)
        with self.lock:
            expired = [key for key, row in self.contexts.items() if now-row['last_time'] > 2.]
            for key in expired:
                self.contexts.pop(key, None)
            if probabilities is None or bbox is None:
                self.contexts.pop(session_id, None)
                return None
            values = np.round(np.asarray(probabilities, dtype=np.float64), 6).astype(np.float32)
            if values.shape != (7,) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError('Expected seven finite, nonnegative probabilities')
            values = np.maximum(values, 1e-6)
            values /= values.sum()
            old = self.contexts.pop(session_id, None)
            history = old['history'] if old else []
            if old and (now <= old['last_time'] or now-old['last_time'] > .3 or box_iou(bbox, old['bbox']) < .3):
                history = []
            history.append((now, values))
            history = [(t, p) for t, p in history[-8:] if now-t <= .85]
            self.contexts[session_id] = {'last_time': now, 'bbox': dict(bbox), 'history': history}
            while len(self.contexts) > self.max_sessions:
                self.contexts.popitem(last=False)
            window = [p for _, p in history]
            window = [window[0]]*(8-len(window))+window
            inputs = np.stack(window)[None]
            logits = self.session.run(None, {'probabilities': inputs})[0].reshape(7)
            scores = np.exp(logits-logits.max())
            scores /= scores.sum()
            if not np.isfinite(scores).all():
                self.contexts.pop(session_id, None)
                raise ValueError('Invalid temporal graph result')
            return scores
