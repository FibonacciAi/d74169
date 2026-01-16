#!/usr/bin/env python3
"""
d74169 Transformer: Breaking the 0.76 Ceiling
=============================================
Can a transformer learn the inverse map: primes → zeros?

The forward map (zeros → primes) is perfect via the explicit formula.
The inverse achieves only 0.76 correlation classically.

Architecture:
- Input: Prime sequence features (positions, gaps, local density)
- Encoder: Transformer with positional encoding
- Output: Predicted Riemann zero positions

Hypothesis: The 0.76 ceiling is due to feature engineering, not fundamental.
A transformer might learn the right representation automatically.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import math
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("@d74169 TRANSFORMER: BREAKING THE 0.76 CEILING")
print("=" * 70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
ZEROS_PATH = '/Users/dimitristefanopoulos/d74169_tests 2.3.3 Path 2/riemann_zeros_master_v2.npy'
try:
    ZEROS = np.load(ZEROS_PATH)
except:
    ZEROS_PATH = '/Users/dimitristefanopoulos/d74169_tests/riemann_zeros_master_v2.npy'
    ZEROS = np.load(ZEROS_PATH)

print(f"Loaded {len(ZEROS)} Riemann zeros")

def sieve(n):
    s = [True] * (n+1)
    s[0] = s[1] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]:
            for j in range(i*i, n+1, i):
                s[j] = False
    return np.array([i for i in range(n+1) if s[i]])

primes = sieve(100000)
print(f"Generated {len(primes)} primes")

# === DATA PREPARATION ===
class PrimeZeroDataset(Dataset):
    """
    Dataset for learning primes → zeros mapping.

    Input: Window of prime features
    Output: Corresponding Riemann zero positions
    """
    def __init__(self, primes, zeros, window_size=64, num_zeros_target=32):
        self.primes = primes
        self.zeros = zeros
        self.window_size = window_size
        self.num_zeros_target = num_zeros_target

        # Precompute prime features
        self.features = self._compute_features()

        # Create samples
        self.samples = self._create_samples()

    def _compute_features(self):
        """Compute rich features for each prime"""
        n = len(self.primes)
        features = np.zeros((n, 8))  # 8 features per prime

        for i, p in enumerate(self.primes):
            # Feature 1: log(p) normalized
            features[i, 0] = np.log(p) / np.log(self.primes[-1])

            # Feature 2: Gap to previous (normalized)
            if i > 0:
                features[i, 1] = (p - self.primes[i-1]) / (2 * np.log(p))

            # Feature 3: Gap to next (normalized)
            if i < n - 1:
                features[i, 2] = (self.primes[i+1] - p) / (2 * np.log(p))

            # Feature 4: Local density (primes in neighborhood)
            low = max(0, i - 10)
            high = min(n, i + 10)
            if self.primes[high-1] > self.primes[low]:
                features[i, 3] = (high - low) / (self.primes[high-1] - self.primes[low]) * np.log(p)

            # Feature 5: Position in prime sequence (normalized)
            features[i, 4] = i / n

            # Feature 6: Residue mod 6
            features[i, 5] = (p % 6) / 6

            # Feature 7: Is twin prime (p-2 or p+2 is prime)
            features[i, 6] = 1.0 if (i > 0 and p - self.primes[i-1] == 2) or \
                                    (i < n-1 and self.primes[i+1] - p == 2) else 0.0

            # Feature 8: Spectral score from first 10 zeros
            log_p = np.log(p)
            gamma = self.zeros[:10]
            features[i, 7] = np.sum(np.cos(gamma * log_p) / np.sqrt(0.25 + gamma**2)) / 10

        return features

    def _create_samples(self):
        """Create training samples"""
        samples = []
        n_primes = len(self.primes)

        # For each window of primes, predict corresponding zeros
        for start_idx in range(0, n_primes - self.window_size, self.window_size // 4):
            end_idx = start_idx + self.window_size
            if end_idx > n_primes:
                break

            # Get prime range
            p_min = self.primes[start_idx]
            p_max = self.primes[end_idx - 1]

            # Target zeros: those with γ roughly in [log(p_min), log(p_max)]
            # Using the asymptotic: N(T) ~ T/(2π) * log(T/(2πe))
            # Inverse: T ~ 2π * N / log(N) approximately

            # Simple heuristic: zeros indexed roughly proportional to prime index
            zero_start = int(start_idx * len(self.zeros) / n_primes)
            zero_end = zero_start + self.num_zeros_target

            if zero_end > len(self.zeros):
                break

            samples.append({
                'prime_features': self.features[start_idx:end_idx],
                'target_zeros': self.zeros[zero_start:zero_end],
                'prime_indices': (start_idx, end_idx),
                'zero_indices': (zero_start, zero_end)
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['prime_features']),
            torch.FloatTensor(sample['target_zeros'])
        )


# === TRANSFORMER MODEL ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PrimeToZeroTransformer(nn.Module):
    """
    Transformer for learning primes → zeros mapping.

    Uses self-attention to capture long-range correlations in prime sequence,
    then decodes to Riemann zero positions.
    """
    def __init__(self, input_dim=8, d_model=128, nhead=8, num_layers=4,
                 num_zeros_output=32, window_size=64):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_zeros_output = num_zeros_output

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: from sequence to fixed-size output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, num_zeros_output)
        )

        # Learnable scale and shift for output
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        # Project to model dimension
        x = self.input_proj(x)  # (batch, seq, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq, d_model)

        # Pool over sequence
        x = x.transpose(1, 2)  # (batch, d_model, seq)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)

        # Project to output zeros
        zeros = self.output_proj(x)  # (batch, num_zeros)

        # Scale and shift
        zeros = zeros * self.output_scale + self.output_shift

        return zeros


# === TRAINING ===
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    """Train the transformer"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    val_corrs = []

    best_corr = -1
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()

                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())

        val_losses.append(val_loss / len(val_loader))

        # Compute correlation
        corr, _ = pearsonr(all_preds, all_targets)
        val_corrs.append(corr)

        if corr > best_corr:
            best_corr = corr
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_losses[-1]:.4f}, "
                  f"Val Loss={val_losses[-1]:.4f}, Corr={corr:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    return train_losses, val_losses, val_corrs, best_corr


# === MAIN ===
print("\n" + "=" * 70)
print("[1] PREPARING DATA")
print("=" * 70)

# Create dataset
dataset = PrimeZeroDataset(primes, ZEROS, window_size=64, num_zeros_target=32)
print(f"Created {len(dataset)} samples")

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# === EXPERIMENT 1: Baseline Transformer ===
print("\n" + "=" * 70)
print("[2] TRAINING BASELINE TRANSFORMER")
print("=" * 70)

model = PrimeToZeroTransformer(
    input_dim=8,
    d_model=128,
    nhead=8,
    num_layers=4,
    num_zeros_output=32,
    window_size=64
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

train_losses, val_losses, val_corrs, best_corr = train_model(
    model, train_loader, val_loader, epochs=100, lr=1e-3
)

print(f"\nBest validation correlation: {best_corr:.4f}")
print(f"The 0.76 ceiling: {'BROKEN!' if best_corr > 0.76 else 'Not broken yet'}")

# === EXPERIMENT 2: Deeper Model ===
print("\n" + "=" * 70)
print("[3] TRAINING DEEPER MODEL (8 layers)")
print("=" * 70)

model_deep = PrimeToZeroTransformer(
    input_dim=8,
    d_model=256,
    nhead=8,
    num_layers=8,
    num_zeros_output=32,
    window_size=64
)

print(f"Deep model parameters: {sum(p.numel() for p in model_deep.parameters()):,}")

train_losses_deep, val_losses_deep, val_corrs_deep, best_corr_deep = train_model(
    model_deep, train_loader, val_loader, epochs=100, lr=5e-4
)

print(f"\nBest deep model correlation: {best_corr_deep:.4f}")

# === ANALYSIS ===
print("\n" + "=" * 70)
print("[4] DETAILED ANALYSIS")
print("=" * 70)

# Get predictions on validation set
model.eval()
model = model.to(device)

all_preds = []
all_targets = []
all_errors = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x)

        preds_np = pred.cpu().numpy()
        targets_np = batch_y.numpy()

        all_preds.extend(preds_np)
        all_targets.extend(targets_np)

        # Per-sample correlation
        for p, t in zip(preds_np, targets_np):
            if len(p) > 2:
                c, _ = pearsonr(p, t)
                all_errors.append(c)

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

print(f"\nPer-sample correlations:")
print(f"  Mean: {np.mean(all_errors):.4f}")
print(f"  Std:  {np.std(all_errors):.4f}")
print(f"  Min:  {np.min(all_errors):.4f}")
print(f"  Max:  {np.max(all_errors):.4f}")

# Global correlation
global_corr, _ = pearsonr(all_preds.flatten(), all_targets.flatten())
print(f"\nGlobal correlation: {global_corr:.4f}")

# === VISUALIZATION ===
print("\n" + "=" * 70)
print("[5] GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('#0a0e1a')

for ax in axes.flat:
    ax.set_facecolor('#131a2e')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#2d3a5a')

fig.suptitle(f'@d74169 Transformer: Primes → Zeros (Best r = {best_corr:.4f})',
             fontsize=16, color='#c4b5fd', fontweight='bold')

# Panel 1: Training curves
ax1 = axes[0, 0]
ax1.plot(train_losses, label='Train', color='#10b981', alpha=0.8)
ax1.plot(val_losses, label='Val', color='#ef4444', alpha=0.8)
ax1.set_xlabel('Epoch', color='#94a3b8')
ax1.set_ylabel('Loss', color='#94a3b8')
ax1.set_title('Training Loss', color='white')
ax1.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 2: Correlation over training
ax2 = axes[0, 1]
ax2.plot(val_corrs, color='#8b5cf6', linewidth=2, label='Baseline')
ax2.plot(val_corrs_deep, color='#f59e0b', linewidth=2, label='Deep')
ax2.axhline(0.76, color='#ef4444', linestyle='--', linewidth=2, label='0.76 ceiling')
ax2.set_xlabel('Epoch', color='#94a3b8')
ax2.set_ylabel('Correlation', color='#94a3b8')
ax2.set_title('Validation Correlation', color='white')
ax2.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 3: Predicted vs Actual zeros
ax3 = axes[1, 0]
sample_idx = 0
pred_sample = all_preds[sample_idx]
target_sample = all_targets[sample_idx]
ax3.scatter(range(len(target_sample)), target_sample, label='Actual', color='#10b981', s=50, alpha=0.7)
ax3.scatter(range(len(pred_sample)), pred_sample, label='Predicted', color='#ef4444', s=50, alpha=0.7, marker='x')
ax3.set_xlabel('Zero Index', color='#94a3b8')
ax3.set_ylabel('γ value', color='#94a3b8')
ax3.set_title('Sample Prediction vs Actual', color='white')
ax3.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 4: Per-sample correlation distribution
ax4 = axes[1, 1]
ax4.hist(all_errors, bins=30, color='#8b5cf6', alpha=0.7, edgecolor='#c4b5fd')
ax4.axvline(0.76, color='#ef4444', linestyle='--', linewidth=2, label='0.76 ceiling')
ax4.axvline(np.mean(all_errors), color='#10b981', linestyle='-', linewidth=2, label=f'Mean={np.mean(all_errors):.3f}')
ax4.set_xlabel('Per-sample Correlation', color='#94a3b8')
ax4.set_ylabel('Count', color='#94a3b8')
ax4.set_title('Correlation Distribution', color='white')
ax4.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = '/Users/dimitristefanopoulos/d74169_tests 2.3.3 Path 2/transformer_inverse.png'
plt.savefig(output_path, dpi=150, facecolor='#0a0e1a', bbox_inches='tight')
print(f"Saved: {output_path}")

# === CONCLUSIONS ===
print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

ceiling_status = "BROKEN" if max(best_corr, best_corr_deep) > 0.76 else "NOT BROKEN"

print(f"""
RESULTS:
  Baseline model (4 layers):  r = {best_corr:.4f}
  Deep model (8 layers):      r = {best_corr_deep:.4f}

  0.76 CEILING STATUS: {ceiling_status}

ANALYSIS:
  - The transformer learns SOME structure in the primes → zeros map
  - Per-sample correlation: mean = {np.mean(all_errors):.4f}, std = {np.std(all_errors):.4f}
  - Some samples achieve high correlation, others fail

POSSIBLE IMPROVEMENTS:
  1. More training data (use more primes/zeros)
  2. Better input features (add more spectral info)
  3. Attention on zeros (decoder with cross-attention)
  4. Curriculum learning (start with easy cases)
  5. Regularization tuning

THE 0.76 CEILING HYPOTHESIS:
  If transformer also hits ~0.76, it suggests:
  - The limit is INFORMATION-THEORETIC, not feature engineering
  - Primes genuinely don't contain enough info to reconstruct zeros
  - The forward map (zeros → primes) loses information irreversibly
""")

# Save model
torch.save(model.state_dict(), '/Users/dimitristefanopoulos/d74169_tests 2.3.3 Path 2/transformer_inverse_model.pt')
print("\nModel saved to transformer_inverse_model.pt")

print("\n[@d74169] Transformer experiment complete.")
