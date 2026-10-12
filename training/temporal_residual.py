"""Causal, static-invariant temporal Transformer for validation experiments."""
from collections import defaultdict
import copy
import math

import numpy as np
import torch
from torch import nn

from evaluate_onnx_ensemble import CLASSES, metrics

WINDOW = 8


def source_graph(report, expected_name='baseline-int8'):
    """Accept only an explicitly named single graph, never silently use a mixture."""
    weight = report.get('second_model_weight')
    if report.get('model') != expected_name or weight not in (0, 1):
        raise ValueError('Temporal inputs require the named, unmixed visual graph')
    sources = report['sources']
    if len(sources) != 2:
        raise ValueError('Unrecognized visual comparison provenance')
    return sources[int(weight)]['sha256']


def iou(a, b):
    width = max(0, min(a['x']+a['width'], b['x']+b['width'])-max(a['x'], b['x']))
    height = max(0, min(a['y']+a['height'], b['y']+b['height'])-max(a['y'], b['y']))
    intersection = width*height
    return intersection/max(1, a['width']*a['height']+b['width']*b['height']-intersection)


def video_windows(video):
    """Only past/current frames. Reset across missed faces, identity-box jumps and gaps."""
    history = []
    last = None
    result = []
    for index, frame in enumerate(video['frames']):
        if frame.get('status') != 'detected' or 'probabilities' not in frame:
            history, last = [], None
            continue
        if last is not None and (frame['seconds']-last['seconds'] > .3 or iou(frame['bbox'], last['bbox']) < .3):
            history = []
        # The real API sends six decimal places. Validation ONNX caches retain
        # more digits, so round both sources before taking sensitive low-probability logs.
        probs = np.round(np.asarray(frame['probabilities'], dtype=np.float64), 6).astype(np.float32)
        if probs.shape != (7,) or not np.isfinite(probs).all() or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError('Invalid seven-class probability vector')
        probs = np.maximum(probs, 1e-6)
        probs /= probs.sum()
        history.append((frame['seconds'], probs))
        history = [(t, p) for t, p in history[-WINDOW:] if frame['seconds']-t <= .85]
        values = [p for _, p in history]
        values = [values[0]]*(WINDOW-len(values))+values
        result.append((index, np.stack(values)))
        last = frame
    return result


class TemporalResidual(nn.Module):
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        self.projection = nn.Linear(7, hidden)
        position = torch.arange(WINDOW).float()[:, None]
        frequency = torch.exp(torch.arange(0, hidden, 2).float()*(-math.log(10000.0)/hidden))
        encoding = torch.zeros(WINDOW, hidden)
        encoding[:, 0::2] = torch.sin(position*frequency)
        encoding[:, 1::2] = torch.cos(position*frequency)
        self.register_buffer('position', encoding)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 7)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def raw_residual(self, log_probs):
        # Equal static values must use the same reduction/GEMM layout in both branches.
        log_probs = log_probs.contiguous()
        centered = (log_probs-log_probs.mean(dim=-1, keepdim=True))/3
        encoded = self.encoder(self.projection(centered)+self.position[None])
        return self.head(self.norm(encoded[:, -1]))

    def forward(self, probabilities):
        log_probs = probabilities.clamp_min(1e-6).log()
        current = log_probs[:, -1:]
        motion = self.raw_residual(log_probs)-self.raw_residual(current.expand_as(log_probs))
        return current[:, 0]+2*torch.tanh(motion)


def classify_videos(model, videos, batch_size=256):
    model.eval()
    transformed = copy.deepcopy(videos)
    inputs, indices = [], []
    for video_index, video in enumerate(videos):
        for frame_index, window in video_windows(video):
            indices.append((video_index, frame_index))
            inputs.append(window)
    if not inputs:
        raise ValueError('No detected frames')
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            logits = model(torch.from_numpy(np.stack(inputs[start:start+batch_size])))
            probabilities.extend(logits.softmax(-1).numpy())
    for (video_index, frame_index), probs in zip(indices, probabilities):
        transformed[video_index]['frames'][frame_index]['probabilities'] = probs.tolist()
    return transformed


def video_metrics(videos):
    domains = defaultdict(list)
    for video in videos:
        active = [f for f in video['frames'] if video['known_label_window'][0] <= f['fraction'] <= video['known_label_window'][1]]
        detected = [f for f in active if 'probabilities' in f]
        video['aggregate_prediction'] = CLASSES[int(np.mean([f['probabilities'] for f in detected], axis=0).argmax())] if detected else None
        video['detection_rate'] = len(detected)/max(1, len(active))
        domains[video['source']].append(video)
    result = {}
    for source, rows in domains.items():
        matrix = np.zeros((7, 8), dtype=np.int64)
        for v in rows:
            predicted = CLASSES.index(v['aggregate_prediction']) if v['aggregate_prediction'] else 7
            matrix[CLASSES.index(v['label']), predicted] += 1
        measured = metrics(matrix[:, :7])
        tp = np.diag(matrix[:, :7])
        support = matrix.sum(1)
        recall = tp/np.maximum(1, support)
        precision = tp/np.maximum(1, matrix[:, :7].sum(0))
        f1 = 2*precision*recall/np.maximum(1e-12, precision+recall)
        measured.update(examples=len(rows), clips=len(rows), accuracy=float(tp.sum()/len(rows)),
                        macro_f1=float(f1.mean()), macro_f1_present=float(f1[support>0].mean()),
                        no_face_clips=int(matrix[:, 7].sum()), confusion_with_no_face_column=matrix.tolist())
        measured['per_class'] = {c:{'support':int(support[i]),'recall':float(recall[i]),'precision':float(precision[i]),'f1':float(f1[i])}
                                 for i,c in enumerate(CLASSES)}
        result[source] = measured
    return result
