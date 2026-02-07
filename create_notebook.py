#!/usr/bin/env python3
"""Create optimized MARec++ Notebook with TURBO_MODE and performance enhancements."""
import json
import os

notebook_path = r'c:\Users\alih1\RS-Project\MARec_Final_Publication.ipynb'

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "accelerator": "GPU"
    },
    "cells": []
}

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src if isinstance(src, list) else [src]}

cells = []

# TITLE
cells.append(md([
    "# MARec++: Metadata Alignment for Cold-Start Recommendation\n",
    "## Reproduction & Enhancements (Optimized)\n",
    "\n",
    "**Paper:** Monteil et al., MARec: RecSys 2024 | **Dataset:** MovieLens HetRec 2011\n",
    "\n",
    "### ⚡ Runtime Modes\n",
    "| Mode | Config | Time |\n",
    "|------|--------|------|\n",
    "| **TURBO** | 1 split × 1 seed | ~5 min |\n",
    "| **FAST** | 3 splits × 2 seeds | ~15 min |\n",
    "| **FULL** | 10 splits × 3 seeds | ~60 min |"
]))

# CONFIG with TURBO_MODE
cells.append(md("---\n## 1. Configuration"))

cells.append(code([
    "import time as _time\n",
    "_NOTEBOOK_START = _time.time()\n",
    "\n",
    "# ========== CHANGE THIS FOR SPEED ==========\n",
    "TURBO_MODE = True   # True = ~5 min, False = ~15 min\n",
    "# ============================================\n",
    "\n",
    "if TURBO_MODE:\n",
    "    CONFIG = {\n",
    "        'seed': 42, 'n_splits': 1, 'n_seeds': 1,\n",
    "        'dataset': 'hetrec', 'cold_train_frac': 0.60, 'cold_val_frac': 0.20,\n",
    "        'lambda1': 1.0, 'alpha': 1.0, 'beta': 100.0, 'delta': 20.0,\n",
    "        'second_order': True, 'fuse_weight': 0.5,\n",
    "        'use_st': True, 'st_model': 'all-MiniLM-L6-v2',\n",
    "        'use_ca_rec': True, 'ca_temperature': 0.07, 'ca_epochs': 5, 'ca_lr': 2e-3, 'ca_hidden_dim': 64,\n",
    "        'use_ua_rec': True, 'ua_epochs': 5, 'ua_lr': 2e-3, 'ua_hidden_dim': 64, 'ua_min_logvar': -10.0, 'ua_max_logvar': 2.0,\n",
    "        'use_ge_rec': True, 'ge_latent_dim': 16, 'ge_hidden_dim': 64, 'ge_epochs': 8, 'ge_lr': 3e-3, 'ge_kl_warmup': 3, 'ge_kl_weight': 0.01,\n",
    "        'ks': [10, 50], 'output_dir': '/content/marec_results',\n",
    "    }\n",
    "    print('⚡ TURBO MODE: 1 split × 1 seed × 4 modes = ~5 min')\n",
    "else:\n",
    "    CONFIG = {\n",
    "        'seed': 42, 'n_splits': 3, 'n_seeds': 2,\n",
    "        'dataset': 'hetrec', 'cold_train_frac': 0.60, 'cold_val_frac': 0.20,\n",
    "        'lambda1': 1.0, 'alpha': 1.0, 'beta': 100.0, 'delta': 20.0,\n",
    "        'second_order': True, 'fuse_weight': 0.5,\n",
    "        'use_st': True, 'st_model': 'all-MiniLM-L6-v2',\n",
    "        'use_ca_rec': True, 'ca_temperature': 0.07, 'ca_epochs': 10, 'ca_lr': 1e-3, 'ca_hidden_dim': 128,\n",
    "        'use_ua_rec': True, 'ua_epochs': 10, 'ua_lr': 1e-3, 'ua_hidden_dim': 128, 'ua_min_logvar': -10.0, 'ua_max_logvar': 2.0,\n",
    "        'use_ge_rec': True, 'ge_latent_dim': 32, 'ge_hidden_dim': 128, 'ge_epochs': 15, 'ge_lr': 2e-3, 'ge_kl_warmup': 5, 'ge_kl_weight': 0.01,\n",
    "        'ks': [10, 25, 50], 'output_dir': '/content/marec_results',\n",
    "    }\n",
    "    print(f'🚀 FAST MODE: {CONFIG[\"n_splits\"]} splits × {CONFIG[\"n_seeds\"]} seeds = ~15 min')"
]))

# ENVIRONMENT with optimizations
cells.append(md("---\n## 2. Environment"))

cells.append(code([
    "import subprocess, sys, os, time, warnings, gc, random, math\n",
    "from collections import defaultdict\n",
    "from itertools import product as iprod\n",
    "\n",
    "def install(pkg): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])\n",
    "for pkg in ['scipy', 'scikit-learn', 'pandas', 'matplotlib', 'seaborn', 'tqdm']: install(pkg)\n",
    "try: import sentence_transformers\n",
    "except ImportError: install('sentence-transformers')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import scipy.sparse as sp\n",
    "from scipy.sparse import csr_matrix\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import MultiLabelBinarizer, normalize\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F_t\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.auto import tqdm\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "def set_seed(seed):\n",
    "    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "set_seed(CONFIG['seed'])\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "USE_FP16 = DEVICE.type == 'cuda'  # Float16 for faster GPU ops\n",
    "print(f'Device: {DEVICE} | FP16: {USE_FP16}')\n",
    "if DEVICE.type == 'cuda':\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
]))

# DATA with progress
cells.append(md("---\n## 3. Data Loading"))

cells.append(code([
    "import urllib.request, zipfile\n",
    "\n",
    "DATA_DIR = '/content/data/hetrec'\n",
    "CACHE_DIR = '/content/cache'\n",
    "os.makedirs(DATA_DIR, exist_ok=True)\n",
    "os.makedirs(CACHE_DIR, exist_ok=True)\n",
    "\n",
    "URL = 'https://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip'\n",
    "\n",
    "def find_file(base, name):\n",
    "    for root, _, files in os.walk(base):\n",
    "        if name in files: return os.path.join(root, name)\n",
    "    return None\n",
    "\n",
    "print('📥 Checking data...')\n",
    "if find_file(DATA_DIR, 'user_ratedmovies.dat') is None:\n",
    "    print('Downloading MovieLens HetRec 2011...')\n",
    "    zp = os.path.join(DATA_DIR, 'hetrec.zip')\n",
    "    urllib.request.urlretrieve(URL, zp)\n",
    "    with zipfile.ZipFile(zp, 'r') as z: z.extractall(DATA_DIR)\n",
    "    os.remove(zp)\n",
    "print('✓ Data ready')"
]))

cells.append(code([
    "# Load interactions\n",
    "rf = find_file(DATA_DIR, 'user_ratedmovies.dat') or find_file(DATA_DIR, 'user_ratedmovies-timestamps.dat')\n",
    "raw = pd.read_csv(rf, sep='\\t', encoding='latin-1')\n",
    "ratings = raw[['userID', 'movieID']].drop_duplicates()\n",
    "\n",
    "all_users = sorted(ratings['userID'].unique())\n",
    "all_items = sorted(ratings['movieID'].unique())\n",
    "user2idx = {u: i for i, u in enumerate(all_users)}\n",
    "item2idx = {it: i for i, it in enumerate(all_items)}\n",
    "idx2item = {i: it for it, i in item2idx.items()}\n",
    "n_users, n_items = len(all_users), len(all_items)\n",
    "\n",
    "row = ratings['userID'].map(user2idx).values\n",
    "col = ratings['movieID'].map(item2idx).values\n",
    "X = csr_matrix((np.ones(len(ratings)), (row, col)), shape=(n_users, n_items))\n",
    "\n",
    "# Load metadata\n",
    "metadata = {}\n",
    "for fname, key, col_name in [('movie_genres.dat', 'genres', 'genre'), ('movie_countries.dat', 'countries', 'country')]:\n",
    "    f = find_file(DATA_DIR, fname)\n",
    "    if f:\n",
    "        df = pd.read_csv(f, sep='\\t', encoding='latin-1')\n",
    "        metadata[key] = df.groupby('movieID')[col_name].apply(list).to_dict()\n",
    "\n",
    "f = find_file(DATA_DIR, 'movie_directors.dat')\n",
    "if f:\n",
    "    df = pd.read_csv(f, sep='\\t', encoding='latin-1')\n",
    "    c = 'directorName' if 'directorName' in df.columns else 'directorID'\n",
    "    metadata['directors'] = df.groupby('movieID')[c].apply(lambda x: [str(v) for v in x]).to_dict()\n",
    "\n",
    "f = find_file(DATA_DIR, 'movies.dat')\n",
    "if f:\n",
    "    try:\n",
    "        df = pd.read_csv(f, sep='\\t', encoding='latin-1')\n",
    "        idc = [c for c in df.columns if 'id' in c.lower()][0]\n",
    "        if 'title' in df.columns: metadata['titles'] = df.set_index(idc)['title'].to_dict()\n",
    "    except: pass\n",
    "\n",
    "print(f'✓ Users: {n_users} | Items: {n_items} | Interactions: {X.nnz}')"
]))

# FEATURES with ST caching
cells.append(md("---\n## 4. Feature Engineering (with ST caching)"))

cells.append(code([
    "def build_feature_matrices(metadata, item2idx, n_items):\n",
    "    fmats = {}\n",
    "    for key in ['genres', 'directors', 'countries']:\n",
    "        if key not in metadata: continue\n",
    "        mlb = MultiLabelBinarizer(sparse_output=True)\n",
    "        labels = [metadata.get(key, {}).get(idx2item.get(i, i), []) for i in range(n_items)]\n",
    "        labels = [[str(v) for v in l] for l in labels]\n",
    "        F = mlb.fit_transform(labels)\n",
    "        fmats[key] = csr_matrix(F)\n",
    "        print(f'  {key}: {F.shape[1]} dims')\n",
    "    return fmats\n",
    "\n",
    "print('🔧 Building features...')\n",
    "fmats = build_feature_matrices(metadata, item2idx, n_items)"
]))

cells.append(code([
    "# Sentence Transformer with CACHING\n",
    "ST_CACHE = os.path.join(CACHE_DIR, 'st_embeddings.npy')\n",
    "\n",
    "if CONFIG['use_st']:\n",
    "    if os.path.exists(ST_CACHE):\n",
    "        print('📂 Loading cached ST embeddings...')\n",
    "        st_emb = np.load(ST_CACHE)\n",
    "    else:\n",
    "        print(f'🧠 Encoding with {CONFIG[\"st_model\"]} (this takes ~3-5 min first time)...')\n",
    "        from sentence_transformers import SentenceTransformer\n",
    "        _st = SentenceTransformer(CONFIG['st_model'])\n",
    "        item_texts = []\n",
    "        for idx in range(n_items):\n",
    "            iid = idx2item.get(idx, idx)\n",
    "            parts = [str(metadata.get('titles', {}).get(iid, ''))]\n",
    "            for k in ['genres', 'directors']:\n",
    "                vals = metadata.get(k, {}).get(iid, [])\n",
    "                if vals: parts.append(', '.join(str(v) for v in vals[:3]))\n",
    "            item_texts.append('. '.join(parts) if parts else 'Unknown')\n",
    "        st_emb = _st.encode(item_texts, show_progress_bar=True, batch_size=128)\n",
    "        np.save(ST_CACHE, st_emb)\n",
    "        del _st; gc.collect()\n",
    "        if torch.cuda.is_available(): torch.cuda.empty_cache()\n",
    "    fmats['st_emb'] = csr_matrix(st_emb)\n",
    "    print(f'✓ ST embeddings: {st_emb.shape}')\n",
    "\n",
    "print(f'✓ Total features: {sum(v.shape[1] for v in fmats.values())} dims')"
]))

# SPLITS
cells.append(code([
    "def create_cold_splits(X, n_splits, seed, train_frac=0.6, val_frac=0.2):\n",
    "    rng = np.random.RandomState(seed); ni = X.shape[1]; splits = []\n",
    "    for _ in range(n_splits):\n",
    "        perm = rng.permutation(ni)\n",
    "        nt, nv = int(ni * train_frac), int(ni * val_frac)\n",
    "        tr, va, te = sorted(perm[:nt].tolist()), sorted(perm[nt:nt+nv].tolist()), sorted(perm[nt+nv:].tolist())\n",
    "        splits.append({'X_train': X[:, tr], 'X_val': X[:, va], 'X_test': X[:, te],\n",
    "                       'train_items': tr, 'val_items': va, 'test_items': te,\n",
    "                       'test_users': np.array(X[:, te].sum(axis=1)).flatten() > 0})\n",
    "    return splits\n",
    "\n",
    "splits = create_cold_splits(X, CONFIG['n_splits'], CONFIG['seed'], CONFIG['cold_train_frac'], CONFIG['cold_val_frac'])\n",
    "print(f'✓ {len(splits)} splits | Train: {len(splits[0][\"train_items\"])} | Test: {len(splits[0][\"test_items\"])} (cold)')"
]))

# MODELS
cells.append(md("---\n## 5. Models"))

cells.append(code([
    "class EASE:\n",
    "    def __init__(self, lambda1=1.0): self.lambda1 = lambda1; self.B = None\n",
    "    def fit(self, X, alignment=None):\n",
    "        n = X.shape[1]; G = (X.T @ X).toarray().astype(np.float64)\n",
    "        XtA = np.zeros_like(G) if alignment is None else (X.T.toarray().astype(np.float64) @ alignment)\n",
    "        P = np.linalg.inv(G + self.lambda1 * np.eye(n) + XtA)\n",
    "        Theta = P @ (G + XtA); dP = np.diag(P).copy(); dP[np.abs(dP) < 1e-10] = 1e-10\n",
    "        self.B = Theta - P * (np.diag(Theta) / dP)[None, :]; return self\n",
    "    def predict(self, X): return (X.toarray() if sp.issparse(X) else X) @ self.B\n",
    "\n",
    "class MARecAligner:\n",
    "    def __init__(self, alpha=1.0, beta=100.0, delta=20.0, second_order=True):\n",
    "        self.alpha, self.beta, self.delta, self.second_order = alpha, beta, delta, second_order\n",
    "        self.mu = self.mu_cross = None; self.names = []\n",
    "    def _d(self, M): return M.toarray() if sp.issparse(M) else np.asarray(M)\n",
    "    def compute_G(self, fmats):\n",
    "        G_list = []; self.names = list(fmats.keys())\n",
    "        for name in self.names:\n",
    "            Fd = self._d(fmats[name]); norms = np.maximum(np.linalg.norm(Fd, axis=1, keepdims=True), 1e-10)\n",
    "            Fn = Fd / norms; Gk = Fn @ Fn.T\n",
    "            sc = (np.abs(Fd) > 0).sum(1).astype(float); mx = max(sc.max(), 1); sf = sc / mx\n",
    "            mask = np.outer(sf, sf); Gk = Gk * mask / (mask + self.delta / (mx + 1)); G_list.append(Gk)\n",
    "        return G_list\n",
    "    def compute_DR(self, Xtr):\n",
    "        clicks = np.array(Xtr.sum(0)).flatten()\n",
    "        p = max(np.percentile(clicks[clicks > 0], 10) if (clicks > 0).any() else 1, 1)\n",
    "        d = np.where(clicks <= p, (self.beta / p) * np.maximum(p - clicks, 0), 0.0); return np.diag(d)\n",
    "    def fit_weights(self, Xtr, G_list):\n",
    "        N = len(G_list); XtX = (Xtr.T @ Xtr).toarray().flatten()\n",
    "        rng = np.random.RandomState(0); idx = rng.choice(len(XtX), min(10000, len(XtX)), replace=False)\n",
    "        xs = XtX[idx]; self.mu = np.ones(N); best = -1e9; grid = [0.0, 1.0, 3.0]\n",
    "        for combo in iprod(grid, repeat=N):\n",
    "            mu = np.array(combo)\n",
    "            if mu.sum() == 0: continue\n",
    "            Gc = sum(mu[k] * G_list[k] for k in range(N)); c = np.corrcoef(Gc.flatten()[idx], xs)[0, 1]\n",
    "            if not np.isnan(c) and c > best: best = c; self.mu = mu.copy()\n",
    "        self.mu_cross = np.zeros((N, N))\n",
    "    def combine_G(self, G_list): return sum(self.mu[k] * G_list[k] for k in range(len(G_list)))\n",
    "    def cross_sim(self, fmats, cold_items, warm_items):\n",
    "        cross = [normalize(self._d(fmats[name][cold_items])) @ normalize(self._d(fmats[name][warm_items])).T for name in self.names]\n",
    "        return sum(self.mu[k] * cross[k] for k in range(len(cross)))\n",
    "\n",
    "print('✓ EASE + MARecAligner')"
]))

cells.append(code([
    "class CARec(nn.Module):\n",
    "    def __init__(self, meta_dim, interact_dim, hidden_dim=64):\n",
    "        super().__init__()\n",
    "        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))\n",
    "        self.interact_proj = nn.Sequential(nn.Linear(interact_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))\n",
    "    def forward(self, mf, vf, temp=0.07):\n",
    "        m = F_t.normalize(self.meta_proj(mf), dim=-1); v = F_t.normalize(self.interact_proj(vf), dim=-1)\n",
    "        return F_t.cross_entropy(m @ v.T / temp, torch.arange(m.shape[0], device=m.device)), m\n",
    "    @torch.no_grad()\n",
    "    def project_meta(self, mf): return F_t.normalize(self.meta_proj(mf), dim=-1)\n",
    "\n",
    "class UARec(nn.Module):\n",
    "    def __init__(self, meta_dim, target_dim, hidden_dim=64):\n",
    "        super().__init__()\n",
    "        self.shared = nn.Sequential(nn.Linear(meta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())\n",
    "        self.fc_mu = nn.Linear(hidden_dim, target_dim); self.fc_logvar = nn.Linear(hidden_dim, target_dim)\n",
    "    def forward(self, mf, targets, min_lv=-10.0, max_lv=2.0):\n",
    "        h = self.shared(mf); mu = self.fc_mu(h); logvar = self.fc_logvar(h).clamp(min_lv, max_lv); var = logvar.exp()\n",
    "        return ((1.0 / var) * (targets - mu).pow(2) + logvar).mean(), mu, var\n",
    "    @torch.no_grad()\n",
    "    def predict(self, mf, min_lv=-10.0, max_lv=2.0):\n",
    "        h = self.shared(mf); return self.fc_mu(h), self.fc_logvar(h).clamp(min_lv, max_lv).exp()\n",
    "\n",
    "class GERec(nn.Module):\n",
    "    def __init__(self, interact_dim, meta_dim, latent_dim=16, hidden_dim=64):\n",
    "        super().__init__()\n",
    "        self.latent_dim = latent_dim\n",
    "        self.encoder = nn.Sequential(nn.Linear(interact_dim + meta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())\n",
    "        self.fc_mu_z = nn.Linear(hidden_dim, latent_dim); self.fc_logvar_z = nn.Linear(hidden_dim, latent_dim)\n",
    "        self.decoder = nn.Sequential(nn.Linear(latent_dim + meta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, interact_dim))\n",
    "    def forward(self, v, m, kl_w=0.01):\n",
    "        h = self.encoder(torch.cat([v, m], -1)); mu_z = self.fc_mu_z(h); logvar_z = self.fc_logvar_z(h)\n",
    "        z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(logvar_z) if self.training else mu_z\n",
    "        v_hat = self.decoder(torch.cat([z, m], -1)); recon = F_t.mse_loss(v_hat, v)\n",
    "        kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())\n",
    "        return recon + kl_w * kl, recon, kl, v_hat\n",
    "    @torch.no_grad()\n",
    "    def generate(self, m): return self.decoder(torch.cat([torch.randn(m.shape[0], self.latent_dim, device=m.device), m], -1))\n",
    "\n",
    "print('✓ CA-Rec, UA-Rec, GE-Rec')"
]))

# EVALUATION
cells.append(code([
    "def evaluate(scores, X_test, ks=(10, 50), user_mask=None):\n",
    "    Xt = X_test.toarray() if sp.issparse(X_test) else X_test\n",
    "    res = {f'hr@{k}': 0.0 for k in ks}; res.update({f'ndcg@{k}': 0.0 for k in ks}); n_eval = 0\n",
    "    for u in range(Xt.shape[0]):\n",
    "        if user_mask is not None and not user_mask[u]: continue\n",
    "        true = np.where(Xt[u] > 0)[0]\n",
    "        if len(true) == 0: continue\n",
    "        ranked = np.argsort(scores[u])[::-1]; ts = set(true)\n",
    "        for k in ks:\n",
    "            topk = ranked[:k]; hits = sum(1 for i in topk if i in ts)\n",
    "            res[f'hr@{k}'] += hits / min(k, len(true))\n",
    "            dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(topk) if i in ts)\n",
    "            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true))))\n",
    "            res[f'ndcg@{k}'] += (dcg / idcg) if idcg > 0 else 0.0\n",
    "        n_eval += 1\n",
    "    for k in res: res[k] /= max(n_eval, 1)\n",
    "    return res\n",
    "print('✓ Evaluation metrics')"
]))

# PIPELINE
cells.append(md("---\n## 6. Pipeline"))

cells.append(code([
    "def get_feats(fmats, items): return sp.hstack([v[items] for v in fmats.values()]).toarray().astype(np.float32)\n",
    "\n",
    "def run_pipeline(split, fmats, cfg, mode='full'):\n",
    "    tr, te = split['train_items'], split['test_items']\n",
    "    Xtr, Xte = split['X_train'], split['X_test']\n",
    "    te_mask = split['test_users']; ks = cfg['ks']\n",
    "    Xtr_d = Xtr.toarray().astype(np.float64); results = {}\n",
    "    \n",
    "    # Baseline\n",
    "    fmats_tr = {k: v[tr] for k, v in fmats.items()}\n",
    "    aligner = MARecAligner(cfg['alpha'], cfg['beta'], cfg['delta'], cfg['second_order'])\n",
    "    G_list = aligner.compute_G(fmats_tr); aligner.fit_weights(Xtr, G_list)\n",
    "    G_comb = aligner.combine_G(G_list); DR = aligner.compute_DR(Xtr)\n",
    "    alignment = aligner.alpha * Xtr_d @ G_comb @ DR\n",
    "    ease = EASE(cfg['lambda1']); ease.fit(Xtr, alignment=alignment)\n",
    "    warm_scores = ease.predict(Xtr); cross_G = aligner.cross_sim(fmats, te, tr)\n",
    "    cold_ease = warm_scores @ cross_G.T; direct = aligner.alpha * Xtr_d @ cross_G.T\n",
    "    baseline = 0.5 * cold_ease + 0.5 * direct\n",
    "    results['MARec Baseline'] = evaluate(baseline, Xte, ks, te_mask)\n",
    "    if mode == 'baseline': return results, baseline\n",
    "    \n",
    "    # Tensors\n",
    "    meta_cold = torch.tensor(get_feats(fmats, te), device=DEVICE)\n",
    "    meta_warm = torch.tensor(get_feats(fmats, tr), device=DEVICE)\n",
    "    XtX = (Xtr.T @ Xtr).toarray().astype(np.float32)\n",
    "    interact_warm = torch.tensor(XtX, device=DEVICE)\n",
    "    meta_dim, interact_dim = meta_cold.shape[1], interact_warm.shape[1]\n",
    "    \n",
    "    # CA\n",
    "    ca_scores = baseline.copy()\n",
    "    if cfg.get('use_ca_rec') and mode in ['+ca', '+ca+ua', 'full']:\n",
    "        ca = CARec(meta_dim, interact_dim, cfg['ca_hidden_dim']).to(DEVICE)\n",
    "        opt = torch.optim.Adam(ca.parameters(), lr=cfg['ca_lr']); ca.train()\n",
    "        for _ in range(cfg['ca_epochs']):\n",
    "            idx = torch.randperm(meta_warm.shape[0], device=DEVICE)[:512]\n",
    "            loss, _ = ca(meta_warm[idx], interact_warm[idx], cfg['ca_temperature'])\n",
    "            opt.zero_grad(); loss.backward(); opt.step()\n",
    "        ca.eval()\n",
    "        ca_sim = ca.project_meta(meta_cold).cpu().numpy() @ ca.project_meta(meta_warm).cpu().numpy().T\n",
    "        ca_scores = 0.6 * baseline + 0.4 * (Xtr_d @ ca_sim.T)\n",
    "        results['+CA-Rec'] = evaluate(ca_scores, Xte, ks, te_mask); del ca, opt\n",
    "        if mode == '+ca': return results, ca_scores\n",
    "    \n",
    "    # UA\n",
    "    ua_scores = ca_scores.copy()\n",
    "    if cfg.get('use_ua_rec') and mode in ['+ca+ua', 'full']:\n",
    "        ua = UARec(meta_dim, interact_dim, cfg['ua_hidden_dim']).to(DEVICE)\n",
    "        opt = torch.optim.Adam(ua.parameters(), lr=cfg['ua_lr']); ua.train()\n",
    "        for _ in range(cfg['ua_epochs']):\n",
    "            idx = torch.randperm(meta_warm.shape[0], device=DEVICE)[:512]\n",
    "            loss, _, _ = ua(meta_warm[idx], interact_warm[idx])\n",
    "            opt.zero_grad(); loss.backward(); opt.step()\n",
    "        ua.eval(); mu_cold, var_cold = ua.predict(meta_cold)\n",
    "        mu_warm, _ = ua.predict(meta_warm)\n",
    "        ua_sim = normalize(mu_cold.cpu().numpy()) @ normalize(mu_warm.cpu().numpy()).T\n",
    "        ua_scores = 0.5 * ca_scores + 0.3 * (Xtr_d @ ua_sim.T) + 0.2 * baseline\n",
    "        results['+CA+UA-Rec'] = evaluate(ua_scores, Xte, ks, te_mask); del ua, opt\n",
    "        if mode == '+ca+ua': return results, ua_scores\n",
    "    \n",
    "    # GE\n",
    "    ge_scores = ua_scores.copy()\n",
    "    if cfg.get('use_ge_rec') and mode == 'full':\n",
    "        ge = GERec(interact_dim, meta_dim, cfg['ge_latent_dim'], cfg['ge_hidden_dim']).to(DEVICE)\n",
    "        opt = torch.optim.Adam(ge.parameters(), lr=cfg['ge_lr']); ge.train()\n",
    "        for ep in range(cfg['ge_epochs']):\n",
    "            kl_w = min(1.0, ep / max(cfg['ge_kl_warmup'], 1)) * cfg['ge_kl_weight']\n",
    "            idx = torch.randperm(meta_warm.shape[0], device=DEVICE)[:512]\n",
    "            loss, _, _, _ = ge(interact_warm[idx], meta_warm[idx], kl_w)\n",
    "            opt.zero_grad(); loss.backward(); opt.step()\n",
    "        ge.eval(); v_cold = ge.generate(meta_cold).cpu().numpy()\n",
    "        ge_sim = normalize(v_cold) @ normalize(XtX).T\n",
    "        ge_scores = 0.4 * ua_scores + 0.35 * (Xtr_d @ ge_sim.T) + 0.25 * baseline\n",
    "        results['+CA+UA+GE-Rec'] = evaluate(ge_scores, Xte, ks, te_mask); del ge, opt\n",
    "    \n",
    "    if torch.cuda.is_available(): torch.cuda.empty_cache()\n",
    "    return results, ge_scores\n",
    "\n",
    "print('✓ Pipeline ready')"
]))

# ABLATION
cells.append(code([
    "def run_ablation(splits, fmats, cfg, seeds):\n",
    "    modes = ['baseline', '+ca', '+ca+ua', 'full']\n",
    "    key_map = {'baseline': 'MARec Baseline', '+ca': '+CA-Rec', '+ca+ua': '+CA+UA-Rec', 'full': '+CA+UA+GE-Rec'}\n",
    "    all_results = {m: defaultdict(list) for m in modes}\n",
    "    total = len(seeds) * len(splits) * len(modes)\n",
    "    pbar = tqdm(total=total, desc='🚀 Ablation')\n",
    "    for seed in seeds:\n",
    "        set_seed(seed)\n",
    "        for split in splits:\n",
    "            for mode in modes:\n",
    "                cfg_run = dict(cfg)\n",
    "                cfg_run['use_ca_rec'] = mode in ['+ca', '+ca+ua', 'full']\n",
    "                cfg_run['use_ua_rec'] = mode in ['+ca+ua', 'full']\n",
    "                cfg_run['use_ge_rec'] = mode == 'full'\n",
    "                res, _ = run_pipeline(split, fmats, cfg_run, mode=mode)\n",
    "                target = key_map[mode]\n",
    "                if target in res:\n",
    "                    for k, v in res[target].items(): all_results[mode][k].append(v)\n",
    "                pbar.update(1)\n",
    "                if torch.cuda.is_available(): torch.cuda.empty_cache()\n",
    "    pbar.close()\n",
    "    summary = {}\n",
    "    for mode in modes:\n",
    "        key = key_map[mode]; summary[key] = {}\n",
    "        for k, vals in all_results[mode].items():\n",
    "            summary[key][k] = float(np.mean(vals)); summary[key][k + '_std'] = float(np.std(vals))\n",
    "    return summary\n",
    "\n",
    "print('✓ Ablation runner ready')"
]))

# RUN
cells.append(md("---\n## 7. Run Experiments"))

cells.append(code([
    "print('=' * 60)\n",
    "print('  RUNNING EXPERIMENTS')\n",
    "n_runs = CONFIG['n_splits'] * CONFIG['n_seeds'] * 4\n",
    "print(f'  {CONFIG[\"n_splits\"]} splits × {CONFIG[\"n_seeds\"]} seeds × 4 modes = {n_runs} runs')\n",
    "print('=' * 60)\n",
    "\n",
    "t0 = time.time()\n",
    "seeds = [CONFIG['seed'] + i * 111 for i in range(CONFIG['n_seeds'])]\n",
    "abl_summary = run_ablation(splits, fmats, CONFIG, seeds)\n",
    "elapsed = time.time() - t0\n",
    "\n",
    "print(f'\\n✓ Done in {elapsed:.0f}s ({elapsed/60:.1f} min)')"
]))

# RESULTS
cells.append(md("---\n## 8. Results"))

cells.append(code([
    "models_order = ['MARec Baseline', '+CA-Rec', '+CA+UA-Rec', '+CA+UA+GE-Rec']\n",
    "\n",
    "print('=' * 70)\n",
    "print(f'{\"Model\":<22s}', end='')\n",
    "for k in CONFIG['ks']: print(f'{\"HR@\" + str(k):>12s}{\"NDCG@\" + str(k):>12s}', end='')\n",
    "print()\n",
    "print('-' * 70)\n",
    "for m in models_order:\n",
    "    if m not in abl_summary: continue\n",
    "    r = abl_summary[m]\n",
    "    print(f'{m:<22s}', end='')\n",
    "    for k in CONFIG['ks']:\n",
    "        print(f'{r.get(f\"hr@{k}\", 0):.4f}±{r.get(f\"hr@{k}_std\", 0):.2f}', end=' ')\n",
    "        print(f'{r.get(f\"ndcg@{k}\", 0):.4f}±{r.get(f\"ndcg@{k}_std\", 0):.2f}', end=' ')\n",
    "    print()\n",
    "print('=' * 70)"
]))

# PAPER COMPARISON
cells.append(code([
    "PAPER = {'MARec (paper)': {'hr@10': 0.2928, 'ndcg@10': 0.3071}}\n",
    "\n",
    "print('\\n📊 Paper Comparison (Table 3):')\n",
    "print(f'  MARec (paper):  HR@10={PAPER[\"MARec (paper)\"][\"hr@10\"]:.4f}')\n",
    "if 'MARec Baseline' in abl_summary:\n",
    "    our = abl_summary['MARec Baseline'].get('hr@10', 0)\n",
    "    lift = (our - PAPER['MARec (paper)']['hr@10']) / PAPER['MARec (paper)']['hr@10'] * 100\n",
    "    print(f'  Our Baseline:   HR@10={our:.4f} ({lift:+.1f}%)')\n",
    "if '+CA+UA+GE-Rec' in abl_summary:\n",
    "    full = abl_summary['+CA+UA+GE-Rec'].get('hr@10', 0)\n",
    "    lift = (full - PAPER['MARec (paper)']['hr@10']) / PAPER['MARec (paper)']['hr@10'] * 100\n",
    "    print(f'  Our Full:       HR@10={full:.4f} ({lift:+.1f}%)')"
]))

# EXPORT
cells.append(md("---\n## 9. Export"))

cells.append(code([
    "import shutil, json as json_mod\n",
    "out = CONFIG['output_dir']; os.makedirs(out, exist_ok=True)\n",
    "rows = [{'model': m, **{k: round(v, 6) for k, v in r.items()}} for m, r in abl_summary.items()]\n",
    "pd.DataFrame(rows).to_csv(os.path.join(out, 'ablation_results.csv'), index=False)\n",
    "shutil.make_archive(out, 'zip', out)\n",
    "total = time.time() - _NOTEBOOK_START\n",
    "print(f'\\n✓ Results saved to {out}.zip')\n",
    "print(f'✓ Total runtime: {total:.0f}s ({total/60:.1f} min)')"
]))

nb['cells'] = cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Created optimized notebook with {len(cells)} cells')
print(f'Location: {notebook_path}')
