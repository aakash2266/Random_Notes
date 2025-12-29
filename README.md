# Jute Pest Classification Model

This repository contains files for a deep learning-based Jute pest image classifier. The model recognizes 17 pest types from image inputs using a Convolutional Neural Network (CNN) trained on the Jute Pest dataset. :contentReference[oaicite:0]{index=0}

## Contents

- `jute_pest_model.h5` — trained Keras model file  
- `class_labels.pkl` — dictionary mapping model output indices to pest labels  
- `predict.py` — script to load the model and make predictions on images  
- Example test image(s) (`test.png`, etc.)

## Dataset

The model was trained on the **UCI Jute Pest Dataset**, which includes images divided into training, validation, and test sets covering 17 pest categories like Beet Armyworm, Black Hairy, Cutworm, Jute Aphid, Yellow Mite, etc. :contentReference[oaicite:1]{index=1}

## Requirements

Install dependencies using:

```bash
pip install tensorflow
pip install numpy
```

*(Optional: Create a virtual environment first.)*

## Usage

1. Put the image you want to classify in the same folder.
2. Update the image file name in `predict.py` (default: `test.png`).
3. Run the prediction script:

```bash
python predict.py
```

The script will print:
- **Predicted pest class**
- **Confidence (% probability)**

## How It Works

- The saved Keras model (`.h5`) contains the CNN architecture and weights.
- The script loads the model and preprocessing steps.
- An input image is resized and normalized before prediction.
- The model outputs probabilities for each of the 17 classes.
- The highest probability becomes the predicted label.

## Example Output

```
Predicted Pest: Black Hairy
Confidence: 87.56 %
```

## Notes

- The model does **not require retraining** at inference time. Simply load and run predictions on new images.
- The `class_labels.pkl` file ensures correct mapping from model outputs to human-readable pest names.

---

If you want to add screenshots or results graphs, you can insert them under the **Usage** or **Example Output** sections following GitHub’s image Markdown syntax. :contentReference[oaicite:2]{index=2}
::contentReference[oaicite:3]{index=3}
