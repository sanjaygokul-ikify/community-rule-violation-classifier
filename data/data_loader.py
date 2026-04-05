import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class RedditCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_data(csv_path, tokenizer, max_length=128, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    X_train, X_val, y_train, y_val = train_test_split(
        df["comment_text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].tolist(),
    )

    train_dataset = RedditCommentDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = RedditCommentDataset(X_val, y_val, tokenizer, max_length)

    return train_dataset, val_dataset
