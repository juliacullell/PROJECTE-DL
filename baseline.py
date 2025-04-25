import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=32, num_classes=30, sampled_features_from_processor: int= 80):
        super().__init__()
        
        # Simple feature extraction
        self.feature_extractor = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Token embedding for previous predictions
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Encoder RNN
        ## It is 80 because is the sample form the Image processor, change it when you change other things
        self.encoder = nn.GRU(sampled_features_from_processor, hidden_dim, batch_first=True)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, prev_tokens, teacher_forcing_ratio=0.5):
        """
        Args:
            x: Input audio features (batch, time, feature)
            prev_tokens: Target tokens for teacher forcing (batch, sequence_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = x.size(0)
        max_len = prev_tokens.size(1)
        
        # Process audio features
        x = F.relu(self.feature_extractor(x))
        
        # Encode the audio sequence
        features, features_hidden = self.encoder(x).relu()
        
        logits = self.classifier(features)
        
        return logits
