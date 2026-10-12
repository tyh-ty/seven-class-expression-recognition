---
language: en
license: apache-2.0
tags:
- vision
- image-classification
- emotion-recognition
- pytorch
- transformers
- vit
datasets:
- EmoSet
- FER2013
- RAF-DB
- RAVDESS
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: jihedjabnoun/vit-face-emotion-recognition
  results:
  - task:
      type: image-classification
      name: Face Emotion Recognition
    metrics:
    - type: accuracy
      value: 0.8295
      name: Accuracy
    - type: f1
      value: 0.8251
      name: Macro F1-Score
    - type: precision
      value: 0.8288
      name: Macro Precision
    - type: recall
      value: 0.8222
      name: Macro Recall
---

# Vision Transformer for Face Emotion Recognition

## Model Description

This model is a fine-tuned Vision Transformer (ViT) for facial emotion recognition. It can classify images into 7 different emotions: Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise.

## Model Details

- **Model Type:** Vision Transformer (ViT)
- **Base Model:** google/vit-base-patch16-224
- **Task:** Image Classification (Emotion Recognition)
- **Languages:** Not applicable (image-based)
- **License:** Apache 2.0

## Training Data

The model was trained on a combination of four popular emotion recognition datasets:

1. **EmoSet**: 21,420 training images, 3,780 validation images
2. **FER2013**: 24,038 training images, 4,235 validation images  
3. **RAF-DB**: 13,038 training images, 2,301 validation images
4. **RAVDESS**: 3,247 training images, 593 validation images

**Total:** 61,743 training images, 10,909 validation images

### Emotion Distribution (Training Set):
- Anger: 8,540 images
- Disgust: 3,760 images  
- Fear: 7,661 images
- Happiness: 14,395 images
- Neutral: 10,114 images
- Sadness: 10,350 images
- Surprise: 6,923 images

## Performance

The model achieves the following performance on the validation set:

- **Overall Accuracy:** 0.8295
- **Macro F1-Score:** 0.8251
- **Macro Precision:** 0.8288
- **Macro Recall:** 0.8222


### Per-Emotion Performance:

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|----------|
| Anger | 0.827 | 0.799 | 0.813 | 1570 |
| Disgust | 0.869 | 0.835 | 0.852 | 629 |
| Fear | 0.786 | 0.743 | 0.764 | 1374 |
| Happiness | 0.926 | 0.914 | 0.920 | 2579 |
| Neutral | 0.767 | 0.820 | 0.793 | 1738 |
| Sadness | 0.760 | 0.798 | 0.779 | 1851 |
| Surprise | 0.867 | 0.846 | 0.856 | 1168 |



### Detailed Classification Report:

```
                precision    recall  f1-score   support

       Anger      0.827    0.799      0.813      1570
     Disgust      0.869    0.835      0.852       629
        Fear      0.786    0.743      0.764      1374
   Happiness      0.926    0.914      0.920      2579
     Neutral      0.767    0.820      0.793      1738
     Sadness      0.760    0.798      0.779      1851
    Surprise      0.867    0.846      0.856      1168

   macro avg      0.829    0.822      0.825     10909
weighted avg      0.831    0.829      0.830     10909
```


## Usage

```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Load model and processor
model = ViTForImageClassification.from_pretrained("jihedjabnoun/vit-face-emotion-recognition")
processor = ViTImageProcessor.from_pretrained("jihedjabnoun/vit-face-emotion-recognition")

# Load and process image
image = Image.open("path_to_your_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

# Get emotion labels
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
predicted_emotion = emotions[predicted_class_id]

# Get confidence scores
probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
confidence = probabilities[predicted_class_id].item()

print(f"Predicted Emotion: {predicted_emotion}")
print(f"Confidence: {confidence:.3f}")

# Show all emotion probabilities
for i, emotion in enumerate(emotions):
    prob = probabilities[i].item()
    print(f"{emotion}: {prob:.3f}")
```

## Training Procedure

### Training Hyperparameters

- **Optimizer:** AdamW
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (training), 32 (evaluation)
- **Number of Epochs:** 4
- **Weight Decay:** 0.01
- **Warmup Steps:** 500

### Hardware

- **GPU:** NVIDIA Tesla T4 (or similar)
- **Training Time:** Approximately 2.5 hours

## Limitations

- The model may have biases inherited from the training datasets
- Performance may vary across different demographic groups
- Images should be clear facial expressions for best results
- The model expects RGB images with faces prominently featured

## Ethical Considerations

- This model should not be used for surveillance or monitoring without proper consent
- Be aware of potential biases in emotion recognition across different demographic groups
- Consider privacy implications when processing facial images

## Citation

If you use this model, please cite:

```bibtex
@misc{jihed-vit-emotion-2025,
  author = {Jihed Jabnoun},
  title = {Vision Transformer for Face Emotion Recognition},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jihedjabnoun/vit-face-emotion-recognition}}
}
```

## Model Card Contact

For questions or feedback about this model, please contact the model author through Hugging Face.
