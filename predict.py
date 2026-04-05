"""
Inference script for the Community Rule Violation Classifier.

Usage:
    python predict.py --comment "Your comment here"
    python predict.py --file data/sample_data.csv
"""

import argparse
import torch
import pandas as pd
from transformers import BertTokenizer

from model.classifier import BertRuleViolationClassifier
from utils.preprocess import clean_text


LABELS = {0: "✅ No Violation", 1: "🚫 Rule Violation"}


def predict_comment(model, tokenizer, text, device, max_length=128):
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(logits, dim=1).item())

    return pred, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="model/best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertRuleViolationClassifier()

    try:
        model.load(args.model_path, device=device)
    except FileNotFoundError:
        print(f"No trained model found at {args.model_path}.")
        print("Please run: python main.py  to train first.\n")
        print("Running with random weights for demonstration...\n")
        model.to(device)

    if args.comment:
        pred, probs = predict_comment(model, tokenizer, args.comment, device)
        print(f"\nComment : {args.comment}")
        print(f"Prediction : {LABELS[pred]}")
        print(f"Confidence : {max(probs)*100:.1f}%")

    elif args.file:
        df = pd.read_csv(args.file)
        results = []
        for _, row in df.iterrows():
            pred, probs = predict_comment(model, tokenizer, row["comment_text"], device)
            results.append({
                "comment": row["comment_text"],
                "prediction": LABELS[pred],
                "confidence": f"{max(probs)*100:.1f}%",
            })
        result_df = pd.DataFrame(results)
        print(result_df.to_string(index=False))
        result_df.to_csv("predictions.csv", index=False)
        print("\nSaved to predictions.csv")

    else:
        # Interactive demo
        print("\nInteractive mode — type a comment and press Enter (Ctrl+C to exit)\n")
        while True:
            try:
                text = input("Comment: ")
                if text.strip():
                    pred, probs = predict_comment(model, tokenizer, text, device)
                    print(f"→ {LABELS[pred]}  ({max(probs)*100:.1f}% confidence)\n")
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    main()
