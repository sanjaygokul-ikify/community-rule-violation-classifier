import torch
import torch.nn as nn
from transformers import BertModel


class BertRuleViolationClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2, dropout=0.3):
        super(BertRuleViolationClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] token representation
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"Model loaded from {path}")
