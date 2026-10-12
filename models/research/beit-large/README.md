
---
license: apache-2.0
tags:
  - emotion-detection
  - affectnet
  - raf-db
  - fer2013
  - beit
  - image-classification
  - facial-expression-recognition
datasets:
  - AffectNet
  - RAF-DB
  - FER-2013
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - confusion matrix
model-index:
  - name: Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large
    results:
      - task:
          type: image-classification
          name: Facial Emotion Classification
        dataset:
          name: AffectNet + RAF-DB + FER-2013
          type: image
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.7357
          - name: Precision
            type: precision
            value: 0.7141
          - name: Recall
            type: recall
            value: 0.6913
          - name: F1
            type: f1
            value: 0.6965
---

# BEiT-Large for Emotion Detection on FER-2013, RAF-DB, and AffectNet

This model is fine-tuned from [`microsoft/beit-large-patch16-224-pt22k-ft22k`](https://huggingface.co/microsoft/beit-large-patch16-224-pt22k-ft22k) for facial emotion recognition using a **combined dataset** made from:

- **FER-2013**
- **RAF-DB**
- **AffectNet (Cleaned & Balanced)**

This fusion of diverse facial expression datasets improves the model's ability to generalize across real-world facial emotion recognition tasks.

---

## 🧠 Classes

The model predicts the following 7 basic emotion classes:

- 😠 **anger**
- 🤢 **disgust**
- 😨 **fear**
- 😀 **happy**
- 😐 **neutral**
- 😢 **sad**
- 😲 **surprise**

---

## 📈 Training Metrics

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1       |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 1     | 0.4788        | 0.6879          | 0.7249   | 0.6906    | 0.6889 | 0.6863   |
| 2     | 0.3865        | 0.6226          | 0.7400   | 0.7005    | 0.7124 | 0.7015   |
| 3     | 0.3277        | 0.6182          | 0.7442   | 0.7042    | 0.7136 | 0.7045   |
| 4     | 0.2713        | 0.6481          | 0.7400   | 0.7025    | 0.7112 | 0.7022   |
| 5     | 0.2372        | 0.7596          | 0.7544   | 0.7263    | 0.7214 | 0.7216   |
| 6     | 0.1916        | 0.7392          | 0.7554   | 0.7234    | 0.7241 | 0.7223   |
| 7     | 0.1569        | 0.8017          | 0.7495   | 0.7179    | 0.7150 | 0.7078   |
| 8     | 0.1297        | 0.8738          | 0.7615   | 0.7306    | 0.7220 | 0.7228   |
| 9     | 0.1129        | 0.9251          | 0.7567   | 0.7283    | 0.7194 | 0.7175   |
| 10    | 0.0987        | 0.8925          | 0.7653   | 0.7337    | 0.7263 | 0.7267   |
| 11    | 0.0819        | 1.0842          | 0.7616   | 0.7398    | 0.7168 | 0.7227   |
| 12    | 0.0671        | 0.9245          | 0.7620   | 0.7332    | 0.7277 | 0.7254   |
| 13    | 0.0591        | 0.9775          | 0.7670   | 0.7400    | 0.7277 | 0.7298   |
| 14    | 0.0523        | 1.0465          | 0.7664   | 0.7449    | 0.7257 | 0.7298   |
| 15    | 0.0476        | 1.0487          | 0.7571   | 0.7274    | 0.7219 | 0.7189   |
| 16    | 0.0481        | 1.1164          | 0.7619   | 0.7414    | 0.7188 | 0.7237   |
| 17    | 0.0378        | 1.1738          | 0.7567   | 0.7376    | 0.7139 | 0.7196   |

✅ **Final Accuracy:** ~**76.2%**  
✅ **Final F1 Score:** ~**0.725**

---

## 🛠️ Training Configuration

The model was trained using the Hugging Face `Trainer` with the following key arguments:

- `num_train_epochs=20`
- `per_device_train_batch_size=32`
- `per_device_eval_batch_size=32`
- `gradient_accumulation_steps=2`
- `learning_rate=2e-5`
- `fp16=True` (mixed precision)
- `eval_strategy="epoch"`
- `save_strategy="epoch"`
- `save_total_limit=2`
- `load_best_model_at_end=True`
- `metric_for_best_model="f1"`
- `greater_is_better=True`

---

## 🔄 Confusion Matrix

![Confusion Matrix](https://huggingface.co/Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large/resolve/main/confusion_matrix.png)

---


## 📊 Classification Report


![Classification Report](https://huggingface.co/Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large/resolve/main/classification_report.png)

---

## 🔧 How to Use

```python
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image

image = Image.open("/your/image/path.jpg").convert("RGB")

processor = BeitImageProcessor.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")
model = BeitForImageClassification.from_pretrained("Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()

print("Predicted emotion:", model.config.id2label[predicted_class])

```

---

## 📄 License

This model is released under the **Apache 2.0 License**. You are free to use, modify, and distribute the model with attribution.

---

## ✍️ Author

- **Username**: Tanneru
- **Base model**: [`microsoft/beit-large-patch16-224-pt22k-ft22k`](https://huggingface.co/microsoft/beit-large-patch16-224-pt22k-ft22k)


---

## 📚 Citation

If you use this model in your work, please cite:

```bibtex
@misc{tanneru2025beit_affectnet,
  title={BEiT-Large fine-tuned on AffectNet for Emotion Detection},
  author={Tanneru},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/Tanneru/Facial-Emotion-Detection-FER-RAFDB-AffectNet-BEIT-Large}},
}

@article{bao2021beit,
  author       = {Hangbo Bao and Li Dong and Furu Wei},
  title        = {BEiT: BERT Pre-Training of Image Transformers},
  journal      = {CoRR},
  volume       = {abs/2106.08254},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.08254},
  archivePrefix = {arXiv},
  eprint       = {2106.08254},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}



