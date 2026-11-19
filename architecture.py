import os
import torch
from torch import nn

class LSTMSingleHead(nn.Module):
    def __init__(self, hidden_size, label_names, dropout):
        super().__init__()
        print("used singlehead classifier model")
        self.classifier = nn.ModuleDict()
        
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            bidirectional=True,
            num_layers=2,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, (hidden_size // 2)),
        nn.ReLU(),
        nn.LayerNorm(hidden_size // 2),
        nn.Dropout(dropout),
        nn.Linear((hidden_size // 2), len(label_names))  # Binary classification for each label (T,6)
        ) # nn.Sequentialは処理が上から下へ順番に流れる場合に，複数の層をまとめて書く関数
        
    def forward(self, hidden_states):
        # logits: (B, T, num_labels)を返す
        # single head
        lstm_outputs, _ = self.lstm(hidden_states)
        logits = self.classifier(lstm_outputs) # (B, T, C)
        return logits
    
class LSTMMultiHead(nn.Module):
    def __init__(self, hidden_size, label_names, dropout):
        super().__init__()
        print("used multihead classifier model")
        self.classifier = nn.ModuleDict()
        self.label_names = label_names
        
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            bidirectional=True,
            num_layers=2,
            batch_first=True
        )
        
        for label in self.label_names:
            self.classifier[label] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)  # Binary classification for each label
            )
        
    def forward(self, hidden_states):
        # logits: (B, T, num_labels)を返す
        # multi head
        logits = {}
        lstm_outputs, _ = self.lstm(hidden_states)
        for label, head in self.classifier.items():
            logits[label] = head(lstm_outputs) # (B, T, 1)
        # Convert logits to a single tensor
        logits = torch.cat([logits[label] for label in self.label_names], dim=2) # Cを1から5に連結, (B,T,C)
        return logits
    

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, label_names, dropout, nhead=8):
        super().__init__()
        print("used ensamble model")
        self.classifier = nn.ModuleDict()
        self.label_names = label_names
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        
        self.linear = nn.Linear(hidden_size, len(label_names))
        
    def forward(self, x):
        output = self.transformer(x)
        logits = self.linear(output)
        return logits
    
class LinearEnsambleHead(nn.Module):
    def __init__(self, hidden_size, label_names, dropout):
        super().__init__()
        print("used ensamble linear model")
        self.classifier = nn.ModuleDict()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(label_names))
        )
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits

class LstmEnsambleHead(nn.Module):
    def __init__(self, hidden_size, label_names, dropout):
        super().__init__()
        print("used ensamble lstm model")
        self.classifier = nn.ModuleDict()
        
        self.lstm = nn.LSTM(
                hidden_size,
                hidden_size // 2,
                bidirectional=True,
                num_layers=1,
                batch_first=True
            )
        
        self.linear = nn.Linear((hidden_size // 2) * 2, len(label_names))
        
    def forward(self, x):
        output, _ = self.lstm(x)
        logits = self.linear(output)
        return logits