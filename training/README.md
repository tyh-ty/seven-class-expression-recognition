# Transformer fine-tuning

This directory contains a standalone training pipeline. It does not modify or integrate with `D:\project_20260718_102644\projects`.

## 1. Prepare the isolated environment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-training-env.ps1
```

The environment is created at `.train-env` with Python 3.11 and CUDA 12.8 PyTorch. It is separate from the ONNX Demo environment.
The setup command also downloads and caches the pretrained ViT-S/16 backbone through the configured Hugging Face mirror.
The command is idempotent: matching CUDA PyTorch packages and cached backbone weights are reused.

## 2A. Prepare the FERPlus benchmark

The current classifier uses seven FERPlus majority-vote labels and excludes contempt. The
preparation command verifies every source file, filters ambiguous/non-face rows
using the FERPlus reference rule, and maps the original splits as follows:

- `Training` -> `train`
- `PublicTest` -> `val`
- `PrivateTest` -> `test`

```powershell
.\.train-env\python.exe .\training\prepare_ferplus.py
```

The generated dataset is stored in `data/ferplus/majority-aligned`. Raw and generated
images are ignored by Git and must not be redistributed without checking the
original FER2013/FERPlus terms for the intended use.

## 2B. Collect natural local corrections

The demo does not require anyone to pose or exaggerate an expression. Use the
camera normally. When a prediction is clearly wrong, click `识别不准` and choose
the visible expression that best matches that frame. Only the current cropped
face is saved; the demo does not automatically record a video or a sequence.

Corrections are stored under `data/custom/feedback/<session>/<label>`. Before
adaptation, prepare them with session-level separation and perceptual duplicate
removal:

```powershell
.\.train-env\python.exe .\training\prepare_feedback_data.py
```

The command ignores sessions whose names start with `_`, including smoke-test
data. By default it refuses to create a dataset until `happiness` and `neutral`
each have at least 20 distinct samples distributed across at least two sessions,
with enough samples in both train and validation. This is intentional: one or a
few corrections are evidence of a problem, but are not enough to retrain safely.

The older prompted collector below remains available for controlled research,
but is not required for the natural correction workflow.

## 2C. Collect prompted research samples

Run the collector once for each subject or capture session. Use keys `1` through
`7` to select an expression, `A` to toggle automatic capture, Space for a
manual frame, and Q to quit. Automatic capture stops at 80 sharp images for the
current label by default. The collector saves the same expanded square face crop
used by the deployed ViT and skips nearly identical consecutive automatic frames.

```powershell
.\.train-env\python.exe .\training\collect_faces.py --subject p001_session01
```

The seven folder labels are `anger`, `disgust`, `fear`, `happiness`, `neutral`, `sadness`, and `surprise`.

Use at least three people for a technical smoke test. A meaningful evaluation requires substantially more people, capture sessions, poses, lighting conditions, ages, and balanced labels.

## 2D. Curate licensed public web images

Public images must be cleaned before training. The curation command validates the
files, removes exact and perceptual duplicates, runs the current ViT as a second
labeling opinion, and creates contact sheets plus a CSV manifest for manual review.

```powershell
.\.train-env\python.exe .\training\curate_web_emotions.py `
  --input .\data\web-public\human-face-emotions\raw\DATASET `
  --output .\data\web-public\human-face-emotions\curated-v4-final `
  --model .\models\vit_emotion_custom.onnx `
  --review-confidence 0 `
  --decisions .\training\review_decisions_human_face_emotions_v1.csv `
  --require-complete-decisions
```

Only `train/` is eligible for training. `review/` and `rejected/` remain excluded.
The included decision file records the manual review of the declared-MIT
`Prasanna18/Human-Face_Images_for_Emotion_Recognition` dataset. Keep the source
README and verify the original source terms before redistribution or production use.

Combine curated web images with the FERPlus training split without changing the
FERPlus validation or test data:

```powershell
.\.train-env\python.exe .\training\combine_training_data.py `
  --base .\data\ferplus\majority-aligned `
  --extra .\data\web-public\human-face-emotions\curated-v4-final\train `
  --output .\data\ferplus\majority-plus-web-core-v1 `
  --extra-repeat 2 `
  --include-extra-classes fear happiness neutral surprise
```

## 3. Split prompted samples by subject

```powershell
.\.train-env\python.exe .\training\split_dataset.py
```

The command refuses to overwrite an existing split. This prevents accidental data loss. It also refuses splits that are missing an emotion class.

## 4. Fine-tune a pure ViT

```powershell
.\.train-env\python.exe .\training\train_transformer.py `
  --data .\data\ferplus\majority-aligned `
  --output .\training\runs\vit-small-ferplus `
  --epochs 30 `
  --batch-size 32
```

The default network is a pretrained ViT-S/16 from `timm`. Selection is based on validation macro F1 rather than accuracy. The default loss applies inverse-square-root class weights at the sample level, which is more stable than raw inverse-frequency weights on rare classes such as disgust. The randomly initialized classifier head uses a higher learning rate than the pretrained backbone so it does not stall during early fine-tuning. The run writes `best.pt`, `history.json`, `best_validation.json`, and `test_report.json`.

When `--initial-checkpoint` is used, its validation score is stored as the epoch 0
baseline. Fine-tuning only replaces `best.pt` after it exceeds that baseline, so an
unsuccessful adaptation run cannot silently replace the stronger starting model.

If GPU memory is insufficient, reduce `--batch-size` or increase `--accumulation-steps`.

## 5. Export and verify ONNX

```powershell
.\.train-env\python.exe .\training\export_onnx.py `
  --checkpoint .\training\runs\vit-small-ferplus\best.pt `
  --output .\models\vit_emotion_custom.onnx
```

The exporter compares PyTorch and ONNX Runtime logits and fails when the numerical difference is too large. A JSON model card is saved beside the ONNX file.

## 6. Adapt to the target webcam

FERPlus contains low-resolution web faces, so a model with a strong benchmark
score can still underperform on a particular camera, room, pose, or population.
Collect consented samples from several capture sessions, split them by session
or subject, and continue fine-tuning from the FERPlus checkpoint:

For natural corrections, add only `feedback-prepared/train` to the benchmark
training split. Keep `feedback-prepared/val` separate as target-camera
validation. Freeze the Transformer backbone for the full adaptation run and
train only the classification head. Do not replace the deployed model unless
both the unchanged FERPlus evaluation and session-isolated target validation
remain at least as strong as the current model.

```powershell
.\.train-env\python.exe .\training\train_transformer.py `
  --data .\data\custom\split `
  --output .\training\runs\vit-small-webcam `
  --initial-checkpoint .\training\runs\vit-small-ferplus-v5\best.pt `
  --epochs 12 `
  --batch-size 16 `
  --learning-rate 1e-5 `
  --head-learning-rate 1e-4 `
  --freeze-backbone-epochs 2
```

Do not report frames from the same capture session as an independent test set.
For a general-purpose model, keep people disjoint across train, validation, and
test. For a personalized model, state clearly that evaluation is user-specific.

Evaluate an exported or quantized ONNX model on the held-out test split:

```powershell
.\.train-env\python.exe .\training\evaluate_onnx.py `
  --model .\models\vit_emotion_custom.onnx `
  --data .\data\ferplus\majority-aligned\test
```

## Evaluation rules

- Keep subjects disjoint across train, validation, and test sets.
- Report macro F1, balanced accuracy, per-class precision/recall/F1, and the confusion matrix.
- Treat expression labels as visible facial patterns, not direct measurements of internal emotion.
- Compare the custom ViT against the unchanged MobileViT and EfficientNet baselines on the same held-out test set.
