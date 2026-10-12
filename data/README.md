# Emotion training data

Real training images are intentionally not committed to the repository.

`training/prepare_ferplus.py` creates the reproducible FERPlus benchmark at
`data/ferplus/majority-aligned`. It uses FERPlus majority-vote labels and the original
FER2013 train/public-test/private-test partition. See `training/README.md` for
the command and source notes.

The collector stores aligned faces by anonymous subject and label:

```text
data/custom/raw/
  p001/
    anger/
    contempt/
    disgust/
    fear/
    happiness/
    neutral/
    sadness/
    surprise/
```

Use different anonymous subject IDs for different people. Dataset splitting is performed by subject, not by image, so near-duplicate frames from one person cannot leak into validation or test data.

Only collect images with informed consent. Do not use the dataset to infer identity, mental health, intent, learning ability, or other sensitive traits.

Public web datasets are stored below `data/web-public/` and are also ignored by
Git. Do not scrape arbitrary image-search results into the training set. Prefer a
dataset with explicit terms, preserve its source README, remove cross-label and
near duplicates, and keep all web images out of the FERPlus validation/test splits.
