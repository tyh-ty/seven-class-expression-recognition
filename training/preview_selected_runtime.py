"""Private local API preview of the frozen candidate, without replacing production."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import server

decision = json.loads((ROOT/'training/overnight-20260927/final-selection.json').read_text(encoding='utf-8'))
graphs = decision['selected']['graphs']
if len(graphs) != 1 or not graphs[0]['grayscale'] or graphs[0]['weight'] != 1:
    raise ValueError('This preview is for the selected single grayscale graph only')
graph = ROOT/graphs[0]['path']
if hashlib.sha256(graph.read_bytes()).hexdigest() != graphs[0]['sha256']:
    raise ValueError('Frozen graph changed')
server.CLASSIFIER_SPECS = (replace(server.CLASSIFIER_SPECS[0], path=graph,
                                 name='Custom ViT-S/16 Seven-Class FP32 (20260928)'),)
server.main()
