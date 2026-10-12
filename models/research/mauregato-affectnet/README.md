---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: vit-base-patch16-224-best-finetuned-on-affectnet_short
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vit-base-patch16-224-best-finetuned-on-affectnet_short

This model is a fine-tuned version of [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9712
- Accuracy: 0.6718
- Precision: 0.6698
- Recall: 0.6718
- F1: 0.6703

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 22

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| 1.9968        | 1.0   | 32   | 1.9113          | 0.2754   | 0.2518    | 0.2754 | 0.2280 |
| 1.4178        | 2.0   | 64   | 1.2704          | 0.5049   | 0.5149    | 0.5049 | 0.4900 |
| 1.1751        | 3.0   | 96   | 1.1116          | 0.5841   | 0.5891    | 0.5841 | 0.5787 |
| 1.0127        | 4.0   | 128  | 1.0237          | 0.6162   | 0.6335    | 0.6162 | 0.6141 |
| 0.9969        | 5.0   | 160  | 0.9890          | 0.6259   | 0.6294    | 0.6259 | 0.6150 |
| 0.9376        | 6.0   | 192  | 0.9768          | 0.6190   | 0.6335    | 0.6190 | 0.6183 |
| 0.8299        | 7.0   | 224  | 0.9579          | 0.6357   | 0.6339    | 0.6357 | 0.6282 |
| 0.7645        | 8.0   | 256  | 0.9366          | 0.6489   | 0.6559    | 0.6489 | 0.6474 |
| 0.7944        | 9.0   | 288  | 0.9303          | 0.6443   | 0.6494    | 0.6443 | 0.6447 |
| 0.7334        | 10.0  | 320  | 0.9510          | 0.6546   | 0.6634    | 0.6546 | 0.6523 |
| 0.6596        | 11.0  | 352  | 0.9369          | 0.6449   | 0.6528    | 0.6449 | 0.6428 |
| 0.6781        | 12.0  | 384  | 0.9717          | 0.6368   | 0.6513    | 0.6368 | 0.6360 |
| 0.5688        | 13.0  | 416  | 0.9509          | 0.6540   | 0.6531    | 0.6540 | 0.6495 |
| 0.5766        | 14.0  | 448  | 0.9485          | 0.6615   | 0.6655    | 0.6615 | 0.6601 |
| 0.5529        | 15.0  | 480  | 0.9590          | 0.6569   | 0.6561    | 0.6569 | 0.6538 |
| 0.4998        | 16.0  | 512  | 0.9677          | 0.6512   | 0.6514    | 0.6512 | 0.6488 |
| 0.4908        | 17.0  | 544  | 0.9670          | 0.6638   | 0.6645    | 0.6638 | 0.6616 |
| 0.4682        | 18.0  | 576  | 0.9635          | 0.6678   | 0.6707    | 0.6678 | 0.6684 |
| 0.4761        | 19.0  | 608  | 0.9680          | 0.6667   | 0.6674    | 0.6667 | 0.6658 |
| 0.4161        | 20.0  | 640  | 0.9701          | 0.6713   | 0.6719    | 0.6713 | 0.6701 |
| 0.4295        | 21.0  | 672  | 0.9712          | 0.6718   | 0.6698    | 0.6718 | 0.6703 |
| 0.434         | 22.0  | 704  | 0.9755          | 0.6707   | 0.6705    | 0.6707 | 0.6690 |


### Framework versions

- Transformers 4.29.0
- Pytorch 2.0.0+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3
