"""
Community Rule Violation Classifier
------------------------------------
BERT-based NLP classifier to detect Reddit comments that violate community guidelines.
Accuracy: ~87%

Usage:
    python main.py --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse
import torch
from transformers import BertTokenizer

from data.data_loader import load_data
from model.classifier import BertRuleViolationClassifier
from model.trainer import train
from utils.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rule Violation Classifier")
    parser.add_argument("--data", type=str, default="data/sample_data.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  Community Rule Violation Classifier — BERT")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    print("Loading data...")
    train_dataset, val_dataset = load_data(
        args.data, tokenizer, max_length=args.max_length
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

    print("Initialising model...")
    model = BertRuleViolationClassifier(model_name=args.model_name)

    print("\nStarting training...\n")
    model = train(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    print("\nEvaluating on validation set...")
    evaluate(model, val_dataset, device=device)


if __name__ == "__main__":
    main()
