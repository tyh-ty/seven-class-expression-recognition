---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{}
---

# Facial-Expression-Recognition
This model is a fine-tuned version of google/vit-base-patch16-224-in21k on the FER 2013 and AffectNet dataset datasets. It achieves the following results on the evaluation set:
  Accuracy - 0.922
  Loss - 0.213
  


### Model Description

The vit-face-expression model is a Vision Transformer fine-tuned for the task of facial emotion recognition.

It is trained on the FER2013and AffectNet datasets, which consist of facial images categorized into eight different emotions:
-anger
-contempt
-sad
-happy
-neutral
-disgust
-fear
-surprise

## Model Details
The model has been fine-tuned using the following hyperparameters:

| Hyperparameter          | Value      |
|-------------------------|------------|
| Train Batch Size        | 32         |
| Eval Batch Size         | 64         |
| Learning Rate           | 2e-4       |
| Gradient Accumulation   | 2          |
| LR Scheduler            | Linear     |
| Warmup Ratio            | 0.04       |
| Num Epochs              | 10         |








## How to Get Started with the Model
Example usage:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

pipe = pipeline("image-classification", model="HardlyHumans/Facial-expression-detection")

processor = AutoImageProcessor.from_pretrained("HardlyHumans/Facial-expression-detection")
model = AutoModelForImageClassification.from_pretrained("HardlyHumans/Facial-expression-detection")

labels = model.config.id2label

outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_label = labels[predicted_class_idx]
```




## Environmental Impact
The net estimated CO2 emission using the  [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute)  scale is around 8.82kg of CO2.



- **Developed by:** Hardly Humans club, IIT Dharwad
- **Model type:** Vision transformer
- **License:** MIT
- **Finetuned from model:** google/vit-base-patch16-224-in21k

- **Hardware Type:** T4 
- **Hours used:** 8+27
- **Cloud Provider:** Google collabotary service
- **Compute Region:** South asia-1
- **Carbon Emitted:** 8.82


### Model Architecture and Objective

