---
tags:
- image-classification
- pytorch
- huggingface
- vit
- emotion-recognition
datasets:
- affectnet
base_model: trpakov/vit-face-expression
library_name: transformers
---

# ViT Face Expression (Fine-tuned on AffectNet)

This model is a fine-tuned version of [trpakov/vit-face-expression](https://huggingface.co/trpakov/vit-face-expression) on the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset.

## Model Description
- **Architecture**: Vision Transformer (ViT)
- **Task**: Facial Emotion Recognition
- **Emotions**: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

## Dataset
AffectNet is a large-scale database of facial expressions in the wild, containing more than 1M facial images from the Internet. This model was fine-tuned on a subset of the manually annotated images covering 7 basic emotions (excluding Contempt to align with the base model's taxonomy).

## Usage

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

repo_name = "michaelgathara/vit-face-affectnet"

processor = ViTImageProcessor.from_pretrained(repo_name)
model = ViTForImageClassification.from_pretrained(repo_name)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 7 emotions
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```
