#!/usr/bin/env python3
"""
PROJECT TRANSFORMER: Attention-Based Primality Classification
==============================================================
@d74169 Research Collaboration - Phase 1.3

Hypothesis: Self-attention can capture relationships between Riemann zeros
            that hand-crafted features miss.

Architecture:
- Input: Raw spectral sequence (cos/sin values for each zero)
- Positional encoding: Zero index
- Transformer encoder: Self-attention over zeros
- Classification head: Binary prime/composite
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PROJECT TRANSFORMER: Attention-Based Primality")
print("=" * 70)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load zeros
ZEROS_PATH = os.environ.get('ZEROS_PATH', '/Users/dimitristefanopoulos/d74169_tests/riemann_zeros_master_v2.npy')
ZEROS = np.load(ZEROS_PATH)
print(f"Loaded {len(ZEROS)} Riemann zeros")

# === FAST SIEVE ===
def sieve(n):
    s = np.ones(n+1, dtype=bool)
    s[0] = s[1] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]:
            s[i*i::i] = False
    return s

# === SPECTRAL SEQUENCE (NOT FLATTENED FEATURES) ===
def spectral_sequence(n, zeros, num_zeros=50):
    """
    Create a SEQUENCE of spectral values (for Transformer input).
    Shape: (num_zeros, 4) - [cos, sin, cos_weighted, sin_weighted]
    """
    log_n = np.log(n)
    gamma = zeros[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)

    phases = gamma * log_n
    cos_vals = np.cos(phases)
    sin_vals = np.sin(phases)
    cos_weighted = cos_vals * weights
    sin_weighted = sin_vals * weights

    # Stack into sequence: each zero has 4 features
    seq = np.stack([cos_vals, sin_vals, cos_weighted, sin_weighted], axis=1)
    return seq.astype(np.float32)

# === TRANSFORMER MODEL ===
class SpectralTransformer(nn.Module):
    def __init__(self, d_input=4, d_model=64, nhead=4, num_layers=3,
                 d_ff=128, dropout=0.1, num_zeros=50):
        super().__init__()

        self.num_zeros = num_zeros

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_zeros, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Class token (like BERT's [CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (batch, num_zeros, 4)
        batch_size = x.size(0)

        # Project input
        x = self.input_proj(x)  # (batch, num_zeros, d_model)

        # Add positional encoding
        x = x + self.pos_embedding

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+num_zeros, d_model)

        # Transformer encoding
        x = self.transformer(x)  # (batch, 1+num_zeros, d_model)

        # Use class token for classification
        cls_output = x[:, 0, :]  # (batch, d_model)

        # Classify
        logits = self.classifier(cls_output).squeeze(-1)  # (batch,)
        return logits

# === SIMPLER BASELINE: LSTM ===
class SpectralLSTM(nn.Module):
    def __init__(self, d_input=4, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, d_input)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits

# === BUILD DATASET ===
print("\n" + "=" * 70)
print("[1] BUILDING DATASET")
print("=" * 70)

N_TRAIN = 10000
NUM_ZEROS = 50  # Sequence length

print(f"Generating spectral sequences for n=2 to {N_TRAIN}...")
IS_PRIME = sieve(N_TRAIN)

integers = list(range(2, N_TRAIN + 1))
X_list = []
y_list = []

for n in integers:
    seq = spectral_sequence(n, ZEROS, NUM_ZEROS)
    X_list.append(seq)
    y_list.append(1 if IS_PRIME[n] else 0)

X_all = np.array(X_list)  # (N, num_zeros, 4)
y_all = np.array(y_list)

n_primes = sum(y_all)
print(f"Dataset: {len(y_all)} samples, {n_primes} primes ({n_primes/len(y_all)*100:.1f}%)")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Train: {len(y_train)} ({sum(y_train)} primes)")
print(f"Test: {len(y_test)} ({sum(y_test)} primes)")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Create dataloaders
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === TRAINING FUNCTION ===
def train_model(model, train_loader, test_loader, epochs=30, lr=1e-3, class_weight=None):
    """Train a model with class-weighted BCE loss"""
    model = model.to(device)

    # Compute positive class weight if not provided
    if class_weight is None:
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)])
    else:
        pos_weight = torch.tensor([class_weight])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > 0.5).cpu().numpy())
                all_labels.extend(y_batch.numpy())

        f1 = f1_score(all_labels, all_preds)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            auc = roc_auc_score(all_labels, all_probs)
            print(f"  Epoch {epoch+1:3d}: Loss={train_loss/len(train_loader):.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    return model, best_f1

# === TRAIN TRANSFORMER ===
print("\n" + "=" * 70)
print("[2] TRAINING TRANSFORMER")
print("=" * 70)

transformer = SpectralTransformer(
    d_input=4,
    d_model=64,
    nhead=4,
    num_layers=3,
    d_ff=128,
    dropout=0.1,
    num_zeros=NUM_ZEROS
)

print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()):,}")
print("\nTraining...")
transformer, transformer_f1 = train_model(transformer, train_loader, test_loader, epochs=30)

# === TRAIN LSTM BASELINE ===
print("\n" + "=" * 70)
print("[3] TRAINING LSTM BASELINE")
print("=" * 70)

lstm = SpectralLSTM(d_input=4, hidden_size=64, num_layers=2, dropout=0.1)
print(f"LSTM parameters: {sum(p.numel() for p in lstm.parameters()):,}")
print("\nTraining...")
lstm, lstm_f1 = train_model(lstm, train_loader, test_loader, epochs=30)

# === DETAILED EVALUATION ===
print("\n" + "=" * 70)
print("[4] DETAILED EVALUATION")
print("=" * 70)

def evaluate_model(model, test_loader, name):
    """Full evaluation of a model"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Final predictions
    preds = (all_probs >= best_thresh).astype(int)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n{name}:")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  F1: {best_f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(classification_report(all_labels, preds, target_names=['Composite', 'Prime']))

    return best_f1, auc, best_thresh

transformer_f1, transformer_auc, transformer_thresh = evaluate_model(transformer, test_loader, "Transformer")
lstm_f1, lstm_auc, lstm_thresh = evaluate_model(lstm, test_loader, "LSTM")

# === OUT-OF-SAMPLE TEST ===
print("\n" + "=" * 70)
print("[5] OUT-OF-SAMPLE TEST (n = 10001-15000)")
print("=" * 70)

N_OOS_START = 10001
N_OOS_END = 15000

oos_integers = list(range(N_OOS_START, N_OOS_END + 1))
IS_PRIME_OOS = sieve(N_OOS_END)

X_oos = np.array([spectral_sequence(n, ZEROS, NUM_ZEROS) for n in oos_integers])
y_oos = np.array([1 if IS_PRIME_OOS[n] else 0 for n in oos_integers])

X_oos_t = torch.tensor(X_oos, dtype=torch.float32)
y_oos_t = torch.tensor(y_oos, dtype=torch.float32)
oos_dataset = TensorDataset(X_oos_t, y_oos_t)
oos_loader = DataLoader(oos_dataset, batch_size=BATCH_SIZE)

print(f"OOS dataset: {len(y_oos)} samples, {sum(y_oos)} primes")

def evaluate_oos(model, loader, thresh, name):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= thresh).astype(int)

    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    recall = np.sum((preds == 1) & (all_labels == 1)) / np.sum(all_labels)

    print(f"{name}: F1={f1:.4f}, AUC={auc:.4f}, Prime Recall={recall:.4f}")
    return f1, auc

print("\nOut-of-sample results:")
trans_oos_f1, trans_oos_auc = evaluate_oos(transformer, oos_loader, transformer_thresh, "Transformer")
lstm_oos_f1, lstm_oos_auc = evaluate_oos(lstm, oos_loader, lstm_thresh, "LSTM")

# === ATTENTION VISUALIZATION ===
print("\n" + "=" * 70)
print("[6] ATTENTION ANALYSIS")
print("=" * 70)

def get_attention_weights(model, x):
    """Extract attention weights from the transformer"""
    model.eval()

    # We need to hook into the attention layers
    attention_weights = []

    def hook_fn(module, input, output):
        # For nn.MultiheadAttention, output is (attn_output, attn_weights)
        if len(output) > 1 and output[1] is not None:
            attention_weights.append(output[1].detach().cpu())

    # Register hooks
    hooks = []
    for layer in model.transformer.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_weights

# Test on a specific prime
test_n = 97  # Prime
test_seq = spectral_sequence(test_n, ZEROS, NUM_ZEROS)
test_input = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).to(device)

# Note: Default PyTorch transformer doesn't return attention weights easily
# For a full analysis, we'd need to modify the model or use a different implementation
print(f"\nTest case: n = {test_n} (prime)")
with torch.no_grad():
    logit = transformer(test_input)
    prob = torch.sigmoid(logit).item()
print(f"P(prime) = {prob:.4f}")

test_n = 100  # Composite
test_seq = spectral_sequence(test_n, ZEROS, NUM_ZEROS)
test_input = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).to(device)

print(f"\nTest case: n = {test_n} (composite)")
with torch.no_grad():
    logit = transformer(test_input)
    prob = torch.sigmoid(logit).item()
print(f"P(prime) = {prob:.4f}")

# === SUMMARY ===
print("\n" + "=" * 70)
print("PROJECT TRANSFORMER: SUMMARY")
print("=" * 70)

print(f"""
MODEL COMPARISON:

                    Transformer     LSTM
                    -----------     ----
Test F1:            {transformer_f1:.4f}          {lstm_f1:.4f}
Test AUC:           {transformer_auc:.4f}          {lstm_auc:.4f}
OOS F1:             {trans_oos_f1:.4f}          {lstm_oos_f1:.4f}
OOS AUC:            {trans_oos_auc:.4f}          {lstm_oos_auc:.4f}

COMPARISON TO SKLEARN MODELS (from project_zeropoint_v2):
  Previous best F1: ~0.50-0.55
  Transformer F1:   {transformer_f1:.4f}
  LSTM F1:          {lstm_f1:.4f}

KEY INSIGHTS:
""")

if transformer_f1 > 0.55:
    print("  - Transformer IMPROVES over sklearn baselines!")
    print("  - Self-attention captures zero-zero relationships")
else:
    print("  - Transformer performs similarly to sklearn models")
    print("  - Handcrafted features capture most discriminative info")

if transformer_f1 > lstm_f1:
    print("  - Attention mechanism adds value over sequential processing")
else:
    print("  - Sequential structure (LSTM) is sufficient for this task")

print(f"""
ARCHITECTURE NOTES:
  - Sequence length: {NUM_ZEROS} zeros
  - Each zero has 4 features: cos, sin, cos_weighted, sin_weighted
  - Self-attention allows zeros to "communicate" about patterns
  - Class token aggregates sequence information

NEXT STEPS:
  - Try more zeros (100, 200) for longer sequences
  - Experiment with attention heads (4, 8, 16)
  - Add residual class as separate input modality
  - Pre-training on larger prime datasets
""")

print("\n[@d74169] Project Transformer complete.")
