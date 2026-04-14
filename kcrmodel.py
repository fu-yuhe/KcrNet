import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.manifold import TSNE
import esm
from pathlib import Path
from typing import Tuple, List, Dict
import time
import pickle
import hashlib
import random
from transformers import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==================== Label Smoothing Loss ====================
class LabelSmoothingBCEWithLogits(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.epsilon) + (1 - targets) * self.epsilon
        loss = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction=self.reduction)
        return loss


# ==================== Configuration ====================
class KcrConfig:
    def __init__(self):
        # Data paths (relative to project root)
        self.data_dir = Path("cleaned_data")
        self.pos_file = self.data_dir / "train_pos.fasta"
        self.neg_file = self.data_dir / "train_neg.fasta"
        self.cache_dir = self.data_dir / "esm2_physchem_ctd_kcr_cache"
        self.sequence_hash_file = self.cache_dir / "sequence_hashes.pkl"
        self.mlm_finetuned_path = self.cache_dir / "esm2_mlm_ft_final.pt"

        # Model settings
        self.model_name = "esm2_t12_35M_UR50D"
        self.sequence_length = 31
        self.center_position = 15

        # Training hyperparameters
        self.batch_size = 34
        self.num_epochs = 60
        self.learning_rate = 0.00075
        self.weight_decay = 6e-5
        self.dropout_rate = 0.34
        self.gru_hidden_size = 128
        self.label_smoothing = 0.08
        self.fixed_threshold = 0.5
        self.early_stopping_patience = 17
        self.early_stopping_min_delta = 0.00075
        self.acc_threshold = 0.838
        self.chunk_size = 1000

        # MLM pretraining settings
        self.mlm_epochs = 5
        self.mlm_batch_size = 8
        self.mlm_learning_rate = 1e-4

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Model Components ====================
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        ch1 = out_channels // 3
        ch2 = out_channels // 3
        ch3 = out_channels - ch1 - ch2
        self.conv3 = nn.Conv1d(in_channels, ch1, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, ch2, 5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, ch3, 7, padding=3)
        self.bn3 = nn.BatchNorm1d(ch1)
        self.bn5 = nn.BatchNorm1d(ch2)
        self.bn7 = nn.BatchNorm1d(ch3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out3 = self.relu(self.bn3(self.conv3(x)))
        out5 = self.relu(self.bn5(self.conv5(x)))
        out7 = self.relu(self.bn7(self.conv7(x)))
        return torch.cat([out3, out5, out7], dim=1)


class EnhancedCBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x).unsqueeze(-1)
        x_channel = x * ca
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x_channel * sa


class KcrPredictor(nn.Module):
    def __init__(self, input_size=480, physchem_dim=5, ctd_dim=21,
                 dropout_rate=0.36, gru_hidden_size=144):
        super().__init__()
        cnn_channels = 76

        self.channel1 = nn.Sequential(
            MultiScaleConvBlock(input_size, cnn_channels),
            EnhancedCBAM(cnn_channels),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.channel2 = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            EnhancedCBAM(cnn_channels),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.bigru = nn.GRU(
            input_size=cnn_channels * 2,
            hidden_size=gru_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.physchem_processor = nn.Sequential(
            nn.Linear(physchem_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.ctd_processor = nn.Sequential(
            nn.Linear(ctd_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        combined_dim = gru_hidden_size * 2 + 16 + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, physchem_features, ctd_features):
        x = x.permute(0, 2, 1)
        x1 = self.channel1(x)
        x2 = self.channel2(x)
        x_combined = torch.cat([x1, x2], dim=1)
        x_combined = self.dropout(x_combined).permute(0, 2, 1)
        gru_out, _ = self.bigru(x_combined)
        gru_out = self.dropout(gru_out)
        attn_weights = self.attention(gru_out)
        context = torch.sum(attn_weights * gru_out, dim=1)

        phys_out = self.physchem_processor(physchem_features)
        ctd_out = self.ctd_processor(ctd_features)

        all_features = torch.cat([context, phys_out, ctd_out], dim=1)
        output = self.classifier(all_features)
        return output


# ==================== Data Loading ====================
class KcrDataLoader:
    @staticmethod
    def _validate_kcr_sequence(seq: str) -> bool:
        return len(seq) == 31 and seq[15] == 'K'

    @staticmethod
    def load_fasta(filepath: str) -> List[str]:
        sequences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        seq = ''.join(current_seq)
                        if KcrDataLoader._validate_kcr_sequence(seq):
                            sequences.append(seq)
                        current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                seq = ''.join(current_seq)
                if KcrDataLoader._validate_kcr_sequence(seq):
                    sequences.append(seq)
        return sequences

    @staticmethod
    def load_dataset(pos_file: str, neg_file: str) -> Tuple[List[str], np.ndarray]:
        print("Loading Kcr dataset...")
        pos_seqs = KcrDataLoader.load_fasta(pos_file)
        neg_seqs = KcrDataLoader.load_fasta(neg_file)
        sequences = pos_seqs + neg_seqs
        labels = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
        print(f"Loaded {len(pos_seqs)} positive, {len(neg_seqs)} negative samples.")
        return sequences, labels


# ==================== Feature Extraction ====================
class PhysChemCTDFeatureExtractor:
    def __init__(self):
        self.hydropathy = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        self.polarity = {
            'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
            'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
            'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0,
            'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9
        }
        self.flexibility = {
            'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346,
            'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462,
            'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509,
            'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386
        }
        self.aa_groups = {
            'hydrophobic': 'AILMFWV',
            'polar': 'NQSTY',
            'positive': 'KRH',
            'negative': 'DE',
            'special': 'CGP'
        }

    def _calc_hydrophobicity(self, seq: str) -> float:
        return sum(self.hydropathy.get(aa, 0) for aa in seq) / len(seq)

    def _calc_charge_density(self, seq: str) -> float:
        pos = sum(1 for aa in seq if aa in 'RKH')
        neg = sum(1 for aa in seq if aa in 'DE')
        return (pos - neg) / len(seq)

    def _calc_polarity(self, seq: str) -> float:
        return sum(self.polarity.get(aa, 0) for aa in seq) / len(seq)

    def _calc_flexibility(self, seq: str) -> float:
        return sum(self.flexibility.get(aa, 0) for aa in seq) / len(seq)

    def _calc_aromaticity(self, seq: str) -> float:
        return sum(1 for aa in seq if aa in 'FWY') / len(seq)

    def extract_physchem(self, sequences: List[str]) -> List[np.ndarray]:
        features = []
        for seq in sequences:
            vec = [
                self._calc_hydrophobicity(seq),
                self._calc_charge_density(seq),
                self._calc_polarity(seq),
                self._calc_flexibility(seq),
                self._calc_aromaticity(seq)
            ]
            features.append(np.array(vec, dtype=np.float32))
        return features

    def extract_ctd(self, sequences: List[str]) -> List[np.ndarray]:
        features = []
        for seq in sequences:
            comp = [sum(1 for aa in seq if aa in group) / len(seq)
                    for group in self.aa_groups.values()]
            # Transition (hydrophobic <-> others)
            hydrophobic = set(self.aa_groups['hydrophobic'])
            trans = 0
            for i in range(len(seq)-1):
                if (seq[i] in hydrophobic) != (seq[i+1] in hydrophobic):
                    trans += 1
            trans /= (len(seq)-1) if len(seq) > 1 else 1
            # Distribution (quartiles)
            dist = []
            for group in self.aa_groups.values():
                pos = [i for i, aa in enumerate(seq) if aa in group]
                if pos:
                    dist.extend([np.percentile(pos, 25)/len(seq),
                                 np.percentile(pos, 50)/len(seq),
                                 np.percentile(pos, 75)/len(seq)])
                else:
                    dist.extend([0, 0, 0])
            ctd_vec = np.concatenate([comp, [trans], dist])
            features.append(ctd_vec.astype(np.float32))
        return features


# ==================== Feature Caching ====================
class ESM2GlobalFeatureCache:
    def __init__(self, config: KcrConfig):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        self.esm2_cache_file = config.cache_dir / "esm2_features.pkl"
        self.physchem_cache_file = config.cache_dir / "physchem_features.pkl"
        self.ctd_cache_file = config.cache_dir / "ctd_features.pkl"

        self.sequence_hashes = self._load_hash()
        self.esm2_cache = self._load_cache(self.esm2_cache_file)
        self.physchem_cache = self._load_cache(self.physchem_cache_file)
        self.ctd_cache = self._load_cache(self.ctd_cache_file)

        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.extractor = PhysChemCTDFeatureExtractor()

    def _load_hash(self):
        if self.config.sequence_hash_file.exists():
            with open(self.config.sequence_hash_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_hash(self):
        with open(self.config.sequence_hash_file, 'wb') as f:
            pickle.dump(self.sequence_hashes, f)

    def _load_cache(self, path):
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def _save_cache(self, cache, path):
        with open(path, 'wb') as f:
            pickle.dump(cache, f)

    def _load_esm_model(self):
        if self.model is None:
            self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            if self.config.mlm_finetuned_path.exists():
                state_dict = torch.load(self.config.mlm_finetuned_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=True)
                print("Loaded MLM fine-tuned weights.")
            self.model = self.model.to(self.config.device)
            self.model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()

    def precompute_all(self, sequences: List[str]):
        print("Checking feature cache...")
        changed = []
        for seq in sequences:
            h = hashlib.md5(seq.encode()).hexdigest()
            if seq not in self.sequence_hashes or self.sequence_hashes[seq] != h:
                changed.append(seq)
                self.sequence_hashes[seq] = h
        self._save_hash()

        if changed:
            print(f"Computing features for {len(changed)} new sequences...")
            self._precompute_esm2(changed)
            self._precompute_global(changed)
        else:
            print("All features are up-to-date.")

    def _precompute_esm2(self, sequences: List[str]):
        self._load_esm_model()
        for i in tqdm(range(0, len(sequences), self.config.chunk_size), desc="ESM2 features"):
            chunk = sequences[i:i+self.config.chunk_size]
            for j in range(0, len(chunk), 8):
                batch = chunk[j:j+8]
                data = [(f"s{k}", seq) for k, seq in enumerate(batch)]
                _, _, tokens = self.batch_converter(data)
                tokens = tokens.to(self.config.device)
                with torch.no_grad():
                    reps = self.model(tokens, repr_layers=[12])["representations"][12].cpu().numpy()
                for k, seq in enumerate(batch):
                    self.esm2_cache[seq] = reps[k].astype(np.float32)
        self._save_cache(self.esm2_cache, self.esm2_cache_file)

    def _precompute_global(self, sequences: List[str]):
        phys = self.extractor.extract_physchem(sequences)
        ctd = self.extractor.extract_ctd(sequences)
        for seq, p, c in zip(sequences, phys, ctd):
            self.physchem_cache[seq] = p
            self.ctd_cache[seq] = c
        self._save_cache(self.physchem_cache, self.physchem_cache_file)
        self._save_cache(self.ctd_cache, self.ctd_cache_file)

    def get_esm2(self, sequences: List[str]) -> List[np.ndarray]:
        return [self.esm2_cache[seq] for seq in sequences]

    def get_physchem(self, sequences: List[str]) -> List[np.ndarray]:
        return [self.physchem_cache[seq] for seq in sequences]

    def get_ctd(self, sequences: List[str]) -> List[np.ndarray]:
        return [self.ctd_cache[seq] for seq in sequences]


# ==================== Dataset & Collate ====================
class KcrDataset(Dataset):
    def __init__(self, esm2_list, phys_list, ctd_list, labels):
        self.esm2 = esm2_list
        self.phys = phys_list
        self.ctd = ctd_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'esm2': torch.FloatTensor(self.esm2[idx]),
            'phys': torch.FloatTensor(self.phys[idx]),
            'ctd': torch.FloatTensor(self.ctd[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def collate_fn(batch):
    esm2 = [item['esm2'] for item in batch]
    phys = torch.stack([item['phys'] for item in batch])
    ctd = torch.stack([item['ctd'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    esm2_padded = torch.nn.utils.rnn.pad_sequence(esm2, batch_first=True)
    return {'esm2': esm2_padded, 'phys': phys, 'ctd': ctd, 'label': labels}


# ==================== Training Utilities ====================
class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0008, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


def calculate_metrics(y_true, y_pred, y_prob):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sp = tn / (tn + fp) if (tn + fp) else 0.0
    except:
        tp = tn = fp = fn = 0
        sp = 0.0
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    mcc_denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-8)
    mcc = (tp*tn - fp*fn) / mcc_denom if mcc_denom != 0 else 0.0
    return {'ACC': acc, 'Precision': pre, 'Recall': rec, 'F1': f1,
            'AUC': auc, 'SP': sp, 'MCC': mcc}


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        esm2 = batch['esm2'].to(device)
        phys = batch['phys'].to(device)
        ctd = batch['ctd'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(esm2, phys, ctd).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            esm2 = batch['esm2'].to(device)
            phys = batch['phys'].to(device)
            ctd = batch['ctd'].to(device)
            logits = model(esm2, phys, ctd).squeeze()
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch['label'].cpu().numpy().tolist())
    probs = np.array(probs)
    preds = (probs > threshold).astype(int)
    return calculate_metrics(labels, preds, probs), probs, labels


def train_model(model, train_loader, val_loader, config, fold_num):
    criterion = LabelSmoothingBCEWithLogits(epsilon=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: min(1.0, (e+1)/8))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=6, min_lr=1e-6)
    early_stop = EarlyStopping(patience=config.early_stopping_patience,
                               min_delta=config.early_stopping_min_delta)
    best_acc = 0
    best_state = None
    for epoch in range(config.num_epochs):
        if epoch < 8:
            warmup.step()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
        metrics, _, _ = evaluate(model, val_loader, config.device, config.fixed_threshold)
        if epoch >= 8:
            scheduler.step(metrics['AUC'])
        if metrics['ACC'] > best_acc:
            best_acc = metrics['ACC']
            best_state = model.state_dict().copy()
        early_stop(metrics['ACC'])
        if early_stop.early_stop:
            print(f"Early stop at epoch {epoch+1}")
            break
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}")
    return best_acc, best_state


# ==================== MLM Pretraining ====================
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_mlm(batch, alphabet):
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([(f"s{i}", seq) for i, seq in enumerate(batch)])
    masked = tokens.clone()
    mask = torch.rand(tokens.shape) < 0.15
    # 80% [MASK]
    rep = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & mask
    masked[rep] = alphabet.mask_idx
    # 10% random
    rand = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & mask & ~rep
    vocab = len(alphabet.all_toks)
    masked[rand] = torch.randint(vocab, tokens.shape)[rand]
    return tokens, masked


def train_mlm(config, sequences):
    print("Starting MLM domain-adaptive pretraining...")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.to(config.device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.mlm_learning_rate)
    loader = DataLoader(SequenceDataset(sequences), batch_size=config.mlm_batch_size,
                        shuffle=True, collate_fn=lambda x: collate_mlm(x, alphabet))
    for epoch in range(config.mlm_epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"MLM Epoch {epoch+1}/{config.mlm_epochs}")
        for orig, masked in pbar:
            orig, masked = orig.to(config.device), masked.to(config.device)
            optimizer.zero_grad()
            logits = model(masked, repr_layers=[])["logits"]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), orig.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item():.4f})
        print(f"MLM Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")
    os.makedirs(config.mlm_finetuned_path.parent, exist_ok=True)
    torch.save(model.state_dict(), config.mlm_finetuned_path)
    print(f"MLM weights saved to {config.mlm_finetuned_path}")


# ==================== Main Pipeline ====================
def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = KcrConfig()
    print(f"Device: {config.device}")

    # Load sequences
    sequences, labels = KcrDataLoader.load_dataset(str(config.pos_file), str(config.neg_file))

    # MLM pretraining if weight not exists
    if not config.mlm_finetuned_path.exists():
        print("MLM weights not found. Starting pretraining...")
        train_mlm(config, sequences)
        # Remove old cache to force recompute
        if config.cache_dir.exists():
            import shutil
            shutil.rmtree(config.cache_dir)
        os.makedirs(config.cache_dir, exist_ok=True)

    # Cache features
    cache = ESM2GlobalFeatureCache(config)
    cache.precompute_all(sequences)

    # Cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    best_acc = 0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences, labels)):
        print(f"\n=== Fold {fold+1} ===")
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_esm2 = cache.get_esm2(train_seqs)
        val_esm2 = cache.get_esm2(val_seqs)
        train_phys = cache.get_physchem(train_seqs)
        val_phys = cache.get_physchem(val_seqs)
        train_ctd = cache.get_ctd(train_seqs)
        val_ctd = cache.get_ctd(val_seqs)

        train_ds = KcrDataset(train_esm2, train_phys, train_ctd, train_labels)
        val_ds = KcrDataset(val_esm2, val_phys, val_ctd, val_labels)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)

        model = KcrPredictor(dropout_rate=config.dropout_rate,
                             gru_hidden_size=config.gru_hidden_size).to(config.device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        fold_acc, fold_state = train_model(model, train_loader, val_loader, config, fold+1)
        results.append(fold_acc)
        if fold_acc > best_acc:
            best_acc = fold_acc
            best_state = fold_state

    if best_state:
        torch.save({'state_dict': best_state, 'config': config.__dict__}, 'kcrnet_best_model.pth')
        print(f"Best model saved with ACC={best_acc:.4f}")

    print(f"5-fold CV average ACC: {np.mean(results):.4f} (+/- {np.std(results):.4f})")


if __name__ == "__main__":
    main()