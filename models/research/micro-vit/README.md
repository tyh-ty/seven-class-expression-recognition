---
license: afl-3.0
datasets:
- LaurenGurgiolo/Micro_Facial_Expressions
language:
- en
metrics:
- accuracy
base_model:
- mo-thecreator/vit-Facial-Expression-Recognition
library_name: transformers
tags:
- facial
- emotion
- recognition
---
"
😐 ViT Micro Facial Expression Recognition Model

This repository contains a Vision Transformer (ViT)–based facial emotion recognition model, iteratively fine-tuned for micro- and macro-facial expression classification.

The model is adapted from mo-thecreator/vit-Facial-Expression-Recognition, a Vision Transformer pre-trained and fine-tuned for facial emotion recognition, and further optimized using a combined dataset of micro-expressions and FER-2013 facial expression data.

📌 Model Details

Base model: mo-thecreator/vit-Facial-Expression-Recognition

Architecture: Vision Transformer (ViT)

Task: Facial Emotion Classification

Output classes: 7 emotion categories

Final model name: vit-micro-facial-expressions

🎭 Emotion Classes

Each facial image is classified into one of the following seven emotion categories:

Label	Emotion
0	Angry
1	Disgust
2	Fear
3	Happy
4	Sad
5	Surprise
6	Neutral
📂 Dataset

The model was trained on a combined dataset consisting of:

Micro Facial Expressions dataset

FER-2013 dataset

Dataset Statistics

Training samples: 28,709 images

Public test samples: 3,589 images

The Micro Facial Expressions dataset is publicly available on Hugging Face:
👉 https://huggingface.co/datasets/LaurenGurgiolo/Micro_Facial_Expressions

🧠 Training Methodology

The base ViT model was iteratively fine-tuned on the combined dataset.

Training focused on improving sensitivity to subtle micro-expressions while maintaining robustness on standard facial expressions.

Iterative fine-tuning enabled progressive refinement of feature representations across emotion classes.

Vision Transformers were selected due to their demonstrated superiority over convolutional neural networks (CNNs) in facial recognition tasks. Empirical studies show that ViTs outperform CNNs in both classification accuracy and generalization capability (Rodrigo et al., 2024).

📊 Performance

Accuracy on micro-expression test dataset: 88%

Evaluation metric: Classification Accuracy

This performance indicates strong generalization for subtle facial expression recognition, particularly in micro-expression scenarios.

🖼️ Training Results Snapshot

Figure 1: Micro-Expressions ViT Model
(Training and evaluation metrics illustrating convergence and performance improvements across epochs.)

🚀 Usage Example
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

processor = AutoImageProcessor.from_pretrained("./vit-micro-facial-expressions")
model = AutoModelForImageClassification.from_pretrained("./vit-micro-facial-expressions")

image = Image.open("face.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_class = outputs.logits.argmax(dim=-1).item()
print(predicted_class)

⚠️ Limitations

Performance may degrade on:

Low-resolution or heavily occluded faces

Extreme head poses or lighting conditions

Emotion labels are inherently subjective and dataset-dependent.

The model is optimized for facial images and may not generalize to non-face imagery.

📜 License & Attribution

Base Model: mo-thecreator/vit-Facial-Expression-Recognition

Datasets: FER-2013 and Micro Facial Expressions dataset licenses apply

Please review the respective Hugging Face dataset and model licenses before commercial use.

🙌 Acknowledgements

Hugging Face for model hosting and datasets

FER-2013 contributors

Micro Facial Expressions dataset authors

Prior research demonstrating ViT effectiveness in facial emotion recognition
"