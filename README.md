# Community Rule Violation Classifier

An NLP model to detect Reddit comments that violate community guidelines, using BERT for text classification.

## Accuracy: 87%

## Features
- Fine-tuned BERT model for binary classification (violation / no violation)
- Preprocessing pipeline for Reddit-style text (URLs, mentions, emojis)
- Training with class imbalance handling via weighted loss
- Evaluation with precision, recall, F1 score, and confusion matrix
- Inference script for single comment or batch prediction

## Tech Stack
- Python 3.9+
- PyTorch
- Hugging Face Transformers (BERT)
- scikit-learn
- pandas, numpy

## Project Structure
```
1-community-rule-violation-classifier/
├── data/
│   ├── sample_data.csv          # Sample labeled dataset
│   └── data_loader.py           # Dataset class
├── model/
│   ├── classifier.py            # BERT classifier model
│   └── trainer.py               # Training loop
├── utils/
│   ├── preprocess.py            # Text cleaning utilities
│   └── evaluate.py              # Metrics and evaluation
├── main.py                      # Train + evaluate pipeline
├── predict.py                   # Run inference on new comments
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python main.py --epochs 3 --batch_size 16 --lr 2e-5
```

### Predict on new comments
```bash
python predict.py --comment "Your comment text here"
```

### Batch prediction
```bash
python predict.py --file data/sample_data.csv
```

## Dataset Format
The model expects a CSV with two columns:
```
comment_text,label
"This is a hateful comment",1
"Great post, thanks!",0
```

Labels: `1` = violation, `0` = no violation

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 87% |
| Precision | 0.85 |
| Recall | 0.88 |
| F1 Score | 0.86 |
