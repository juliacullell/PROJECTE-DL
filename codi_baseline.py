# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 21:07:46 2025

@author: nildi
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import librosa
from transformers import Speech2TextProcessor
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Configuració inicial
# -------------------------------
audio_root = './emotifymusic/'  # carpeta que conté classical/, rock/, etc.
model_name = "facebook/s2t-small-librispeech-asr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Dataset personalitzat sense CSV
# -------------------------------
class MusicGenreFolderDataset(Dataset):
    def __init__(self, audio_root, processor, target_sr=16000):
        self.samples = []
        self.labels = []
        self.processor = processor
        self.target_sr = target_sr

        # Explorar carpetes (una per gènere)
        for genre in os.listdir(audio_root):
            genre_path = os.path.join(audio_root, genre)
            if not os.path.isdir(genre_path):
                continue

            for fname in os.listdir(genre_path):
                if fname.endswith('.mp3'):
                    self.samples.append(os.path.join(genre_path, fname))
                    self.labels.append(genre)

        # Codificar els noms de carpeta com a etiquetes
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        waveform = waveform.squeeze()
        waveform = waveform / waveform.abs().max()

        trimmed, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
        waveform = torch.tensor(trimmed)

        inputs = self.processor(waveform, sampling_rate=self.target_sr, return_tensors="pt")
        input_features = inputs.input_features.squeeze(0)  # (seq_len, features)

        label_encoded = self.label_encoder.transform([label])[0]
        return input_features, label_encoded

# -------------------------------
# 3. Inicialitzar dades
# -------------------------------
print("Carregant processor...")
processor = Speech2TextProcessor.from_pretrained(model_name)

print("Creant dataset...")
dataset = MusicGenreFolderDataset(audio_root, processor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    inputs, labels = zip(*batch)
    # Padding seqüències a la mateixa llargada
    padded_inputs = pad_sequence(inputs, batch_first=True)  # shape: (batch, max_len, 80)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_inputs, labels

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)


# -------------------------------
# 4. Model Baseline (el teu model)
# -------------------------------




class Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=32, num_classes=4, sampled_features_from_processor=80):
        super().__init__()
        self.feature_extractor = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, prev_tokens, teacher_forcing_ratio=0.5):
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = torch.relu(self.feature_extractor(x))  # (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        features, _ = self.encoder(x)
        logits = self.classifier(features)
        return logits

input_dim = 80
num_classes = len(dataset.label_encoder.classes_)
model = Baseline(input_dim=input_dim, num_classes=num_classes).to(device)

# -------------------------------
# 5. Entrenament
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Entrenant...")

for epoch in range(2):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print("Input features shape:", inputs.shape)  # Inputs abans d'entrar al model
        print("Input features example:", inputs[0])   # Primer exemple del batch
        optimizer.zero_grad()
        outputs = model(inputs, prev_tokens=torch.zeros_like(labels).unsqueeze(1))
        print("Output logits shape:", outputs.shape)  # Sortida del model
        print("Output logits example:", outputs[0])   # Primer exemple de sortida
        outputs = outputs.mean(dim=1)  # mitjana sobre seqüència
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        break  #Aquest break serveix per a que no estigui printejant tot el rato els vectors, per a calcular la loss real, millor treure'l

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Entrenament completat!")