
# -------------------------------------------------- 1. Full Pipeline: Feature Update + Prediction Tasks --------------------------------------------------
import re
import random
from pathlib import Path
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from lifelines.utils import concordance_index

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, global_mean_pool


# -------------------------------------------------- 2. Global Cache & Path Configuration --------------------------------------------------
# Whether to enable feature update
ENABLE_FEATURE_UPDATE = True

# Result output root directory
result_root = Path("../Results/")
result_root.mkdir(parents=True, exist_ok=True)

epoch_data_dir = result_root / "epoch_metrics"
epoch_data_dir.mkdir(parents=True, exist_ok=True)

# Model hyperparameters
seed = 20250715
batch_size = 64
lr_init = 0.001
epochs = 30
hidden_dim = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_n = 20
cosine_thres = 0.3

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -------------------------------------------------- 3. Feature Definitions --------------------------------------------------
stats = ['ID', 'BCLC A', 'BCLC B', 'BCLC C', 'Age', 'Male', 'HBV', 'HCV']

t_base = ['resection', 'ablation', 'TACE', 'HAIC', 'systemic_treatment']
fs = ['ViableLesion', 'NewLesion', 'Amount=2/3', 'Amount>3', 'Diameter=50-70', 'Diameter>70',
      'Metastasis', 'LLNM', 'Varicosity', 'VI', 'AFP=400-800', 'AFP>800', 'CPS=B', 'CPS=C', 'ALBI=2', 'ALBI=3']

f1 = [f'1st_{feat}' for feat in fs[2:]]
t1 = [f'1st_{t}' for t in t_base]
t2 = [f'2nd_{t}' for t in t_base]

prefixes = ['2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']
f2, f3, f4, f5, f6, f7, f8, f9 = [[f"{p}_{feat}" for feat in fs] for p in prefixes]

survival_outcome = ['is_death', 'duration']
binary_outcome = ['5year-S']


# -------------------------------------------------- 4. Data Processing Functions --------------------------------------------------
def discrete(data):
    """Discretize continuous clinical variables into categorical features."""

    # 1. BCLC Stage
    data['BCLC A'] = data['BCLC'].isin(['Z', 'A']).astype(int)
    for stage in ['B', 'C', 'D']:
        data[f'BCLC {stage}'] = (data['BCLC'] == stage).astype(int)

    # 2. Process each time-step
    prefixes = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']

    for p in prefixes:
        check_col = f'{p}_Amount' if p == '1st' else f'{p}_ViableLesion'
        if check_col not in data.columns:
            continue

        # Amount
        amt = data[f'{p}_Amount']
        data[f'{p}_Amount=1'] = (amt <= 1).astype(int)
        data[f'{p}_Amount=2/3'] = amt.between(1.1, 3).astype(int)
        data[f'{p}_Amount>3'] = (amt > 3).astype(int)

        # Diameter
        dia = data[f'{p}_Diameter']
        data[f'{p}_Diameter<50'] = (dia < 50).astype(int)
        data[f'{p}_Diameter=50-70'] = dia.between(50, 70).astype(int)
        data[f'{p}_Diameter>70'] = (dia > 70).astype(int)

        # AFP
        afp = data[f'{p}_AFP']
        data[f'{p}_AFP<400'] = (afp < 400).astype(int)
        data[f'{p}_AFP=400-800'] = afp.between(400, 800).astype(int)
        data[f'{p}_AFP>800'] = (afp > 800).astype(int)

        # CPS
        cps = data[f'{p}_CPS']
        data[f'{p}_CPS=A'] = (cps <= 6).astype(int)
        data[f'{p}_CPS=B'] = cps.between(6.1, 9).astype(int)
        data[f'{p}_CPS=C'] = (cps >= 10).astype(int)

        # ALBI
        albi = data[f'{p}_ALBI']
        for i in [1, 2, 3]:
            data[f'{p}_ALBI={i}'] = (albi == i).astype(int)

        # Systemic Treatment
        target_col = f'{p}_targted_drug'
        immuno_col = f'{p}_immunodrug'
        if target_col in data.columns and immuno_col in data.columns:
            data[f'{p}_systemic_treatment'] = data[[target_col, immuno_col]].max(axis=1)

    return data


def select_cols(data):
    """Select required columns based on available time-step features."""
    data = data.loc[data['BCLC'] != 'D',].reset_index().drop(['index'], axis=1)

    # Base columns (always included)
    base = survival_outcome + binary_outcome + stats + f1 + t1

    # Check each stage from highest to lowest
    stage_check = [
        ('9th_ViableLesion', f9), ('8th_ViableLesion', f8),
        ('7th_ViableLesion', f7), ('6th_ViableLesion', f6),
        ('5th_ViableLesion', f5), ('4th_ViableLesion', f4),
        ('3rd_ViableLesion', f3),
    ]

    # Collect available feature groups (from high to low stage)
    available = []
    for check_col, fgroup in stage_check:
        if check_col in data.columns:
            available.append(fgroup)

    # Reverse to get low-to-high order (f3, f4, ..., f9)
    available = available[::-1]

    # Check f2/t2 stage
    if '2nd_systemic_treatment' in data.columns:
        base += f2 + t2
    elif '2nd_ViableLesion' in data.columns:
        base += f2

    # Append all available higher-stage features
    for fg in available:
        base += fg

    return data[base]


def data_split(df, label_col):
    """Split data into train, validation, and test sets."""
    y_event = df[label_col].values

    # Split out 20% as test set, 80% as train+validation
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(df, y_event))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Split 15% from train+validation as validation set
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_df, trainval_df[label_col]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# -------------------------------------------------- 5. Feature Update Module --------------------------------------------------
class FeatureUpdateGNN(nn.Module):
    """GNN model for feature update via graph-based message passing."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Residual connection weight

    def forward(self, x, edge_index):
        x_conv = self.conv1(x, edge_index)
        x_conv = F.relu(x_conv)
        x_conv = F.dropout(x_conv, p=0.2, training=self.training)

        x_conv = self.conv2(x_conv, edge_index)
        x_conv = F.relu(x_conv)

        x_updated = self.decoder(x_conv)
        x_final = self.alpha * x_updated + (1 - self.alpha) * x  # Residual connection

        return x_final


def cox_ph_loss(risk_scores, durations, events, eps=1e-8):
    """
    Numerically stable Cox Proportional Hazards loss.
    Prevents NaNs by clamping risk scores and adding epsilon.
    """
    risk_scores = torch.clamp(risk_scores, min=-20, max=20)
    exp_risk = torch.exp(risk_scores)

    # Sort by duration descending for efficient risk set calculation
    indices = torch.argsort(durations, descending=True)
    exp_risk = exp_risk[indices]
    events = events[indices]

    # Cumulative sum of risk (from longest to shortest duration)
    risk_set_sum = torch.cumsum(exp_risk, dim=0) + eps

    # Negative log partial likelihood (only for uncensored events)
    log_loss = torch.log(risk_set_sum) - risk_scores[indices]
    return torch.sum(log_loss * events) / (torch.sum(events) + eps)


def build_node_graph(df, neighbor_vectors, cosine_threshold=0.5, max_neighbors=20, task_name=None):
    """Build a node-level homogeneous graph for feature update based on cosine similarity."""
    global neighbor_cache

    n_samples = len(df)

    # Extract and standardize neighbor computation vectors
    X_neighbor = df[neighbor_vectors].values
    X_neighbor = StandardScaler().fit_transform(X_neighbor)

    # Compute cosine distance matrix
    cosine_dist_matrix = cosine_distances(X_neighbor)
    print(f"Neighbor vector shape: {X_neighbor.shape}")
    print(f"Cosine distance range: [{cosine_dist_matrix.min():.4f}, {cosine_dist_matrix.max():.4f}]")

    # Exclude self-connections
    np.fill_diagonal(cosine_dist_matrix, 2.0)

    # Build edge index
    edges = set()
    neighbor_counts = []

    for i in range(n_samples):
        candidate_neighbors = np.where(cosine_dist_matrix[i] < cosine_threshold)[0]
        candidate_distances = cosine_dist_matrix[i][candidate_neighbors]
        sorted_indices = np.argsort(candidate_distances)
        sorted_neighbors = candidate_neighbors[sorted_indices]
        selected_neighbors = sorted_neighbors[:max_neighbors]
        neighbor_counts.append(len(selected_neighbors))

        # Add undirected edges
        for j in selected_neighbors:
            if i < j:
                edges.add((i, j))
                edges.add((j, i))

    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # Prepare labels
    y_is_death = torch.tensor(df['is_death'].values, dtype=torch.float)
    y_duration = torch.tensor(df['duration'].values, dtype=torch.float)
    y_5year = torch.tensor(df['5year-S'].values, dtype=torch.float)

    data = Data(
        x=torch.tensor(X_neighbor, dtype=torch.float),
        edge_index=edge_index,
        y_is_death=y_is_death,
        y_duration=y_duration,
        y_5year=y_5year
    )

    return data, cosine_dist_matrix, neighbor_counts


def update_features_with_gnn(df, neighbor_vectors, update_features, task_name,
                             model=None, scaler=None, cosine_threshold=0.5,
                             max_neighbors=20, train_model=True, dataset_name="train"):
    """Update patient features using GNN-based graph message passing."""

    if not ENABLE_FEATURE_UPDATE:
        return _handle_disabled_update(df, neighbor_vectors, update_features)

    # Setup Graph
    graph_data, *_ = build_node_graph(df, neighbor_vectors, cosine_threshold, max_neighbors, task_name)

    if len(df) != graph_data.x.size(0):
        raise ValueError(f"Mismatch: {len(df)} rows vs {graph_data.x.size(0)} nodes.")

    # Scaling
    scaler = scaler or StandardScaler().fit(df[neighbor_vectors].values)
    graph_data.x = torch.from_numpy(scaler.transform(df[neighbor_vectors].values)).float()

    # Model initialization
    model = (model or FeatureUpdateGNN(len(neighbor_vectors), 64)).to(device)

    # Track training health
    is_healthy = True
    if train_model:
        print(f"🚀 Training {dataset_name} features...")
        is_healthy = _train_feature_model(model, graph_data.to(device), task_name)

    # Inference
    model.eval()
    model.cpu()
    with torch.no_grad():
        out = model(graph_data.cpu().x, graph_data.cpu().edge_index).numpy()
        updated_values = scaler.inverse_transform(out)

    # Map updated features back to DataFrame
    updated_df = df.copy()
    feat_map = {feat: i for i, feat in enumerate(neighbor_vectors)}
    for feat in filter(lambda f: f in feat_map, update_features):
        updated_df[feat] = updated_values[:, feat_map[feat]]

    return updated_df, graph_data, model, scaler, is_healthy


def _handle_disabled_update(df, neighbor_vectors, update_features):
    """Pass-through handler when feature update is disabled."""
    print("⏩ Feature update disabled. Using raw data.")
    mock_graph = Data(
        x=torch.from_numpy(df[neighbor_vectors].values).float(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        **{f'y_{k}': torch.from_numpy(df[v].values).float()
           for k, v in [('is_death', 'is_death'), ('duration', 'duration'), ('5year', '5year-S')]}
    )
    return df.copy(), mock_graph, None, None, True  # True = healthy


def _train_feature_model(model, data, task_name, epochs=10, patience=5):
    """Train the feature update GNN with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss, best_state, counter = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        try:
            out = model(data.x, data.edge_index)
            loss = cox_ph_loss(out.mean(dim=1), data.y_duration, data.y_is_death)

            if torch.isnan(loss): raise ValueError("NaN Loss")

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss, counter = loss.item(), 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
            if counter >= patience: break

        except (ValueError, RuntimeError):
            print(f"⚠️ NaN detected in {task_name}. Rolling back to best state.")
            if best_state:
                model.load_state_dict(best_state)
            else:
                _load_last_valid_from_disk(model, task_name)
            return False  # Interrupted

    return True  # Completed normally


def _load_last_valid_from_disk(model, task_name):
    """Fallback: load last valid weights from disk if in-memory state is lost."""
    path = epoch_data_dir / f"{task_name}_last_valid_weights.pth"
    if path.exists():
        model.load_state_dict(torch.load(path))


# -------------------------------------------------- 6. Prediction Task Module --------------------------------------------------
class GNN_Survival(nn.Module):
    """Heterogeneous GNN model for survival analysis with GAT-based attention."""

    def __init__(self, sample, hidden=128, heads=8):
        super().__init__()
        # Linear projection per node type
        self.lin = nn.ModuleDict({
            nt: nn.Linear(sample[nt].num_features, hidden)
            for nt in sample.node_types
        })

        # Attention-based heterogeneous graph convolution (self-loops disabled)
        self.conv1 = HeteroConv({
            et: GATConv(
                (-1, -1),
                hidden,
                heads=heads,
                concat=False,
                add_self_loops=False
            )
            for et in sample.edge_types
        }, aggr='mean')

        self.conv2 = HeteroConv({
            et: GATConv(
                (-1, -1),
                hidden,
                heads=heads,
                concat=False,
                add_self_loops=False
            )
            for et in sample.edge_types
        }, aggr='mean')

        # Output layer
        self.fc = nn.Linear(hidden * len(sample.node_types), 1)

    def forward(self, data):
        x = {nt: F.relu(self.lin[nt](data[nt].x)) for nt in data.node_types}
        x = self.conv1(x, data.edge_index_dict)
        x = {k: F.relu(v) for k, v in x.items()}
        x = self.conv2(x, data.edge_index_dict)

        pooled = [global_mean_pool(x[nt], data[nt].batch) for nt in data.node_types]
        g_emb = torch.cat(pooled, dim=1)
        return self.fc(g_emb).squeeze(-1)


def ci_with_ci(risk, time, event, n_boot=1000):
    """Compute C-index with bootstrap 95% confidence interval."""
    ci = concordance_index(time, -risk, event)
    idx = np.arange(len(risk))
    boot = []
    for _ in range(n_boot):
        samp = np.random.choice(idx, size=len(idx), replace=True)
        if len(np.unique(event[samp])) < 2:
            continue
        boot.append(concordance_index(time[samp], -risk[samp], event[samp]))
    if len(boot) >= 10:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo, hi = ci, ci
    return ci, lo, hi


def evaluate_Cindex(loader, model, df=None, task_name=None):
    """Evaluate model on a data loader and return C-index with confidence interval."""
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            risks.append(model(batch).cpu())
            times.append(batch.y_time.cpu())
            events.append(batch.y_event.cpu())

    risk = torch.cat(risks).numpy()
    time = torch.cat(times).numpy()
    event = torch.cat(events).numpy()

    # Check for NaN values
    if np.isnan(risk).any() or np.isnan(time).any() or np.isnan(event).any():
        nan_risk = np.isnan(risk).sum()
        nan_time = np.isnan(time).sum()
        raise ValueError(f"NaNs detected in inputs. Risk NaN: {nan_risk}, Time NaN: {nan_time}")

    # Compute C-index with confidence interval
    ci, lo, hi = ci_with_ci(risk, time, event)

    if df is not None:
        result_df = df.copy()
        result_df['survival_risk'] = risk
        return (ci, lo, hi), result_df, time, event

    return (ci, lo, hi), risk


def train_survival_model(train_loader, val_loader, test_loader,
                         train_df, val_df, test_df,
                         sample_graph, task_name, original_features):
    """Train the survival analysis GNN model with early stopping and best-model checkpointing."""
    print(f"Starting survival model training: {task_name}")

    # Ensure task directory exists for saving .pt files
    task_dir = epoch_data_dir / task_name
    task_dir.mkdir(exist_ok=True)

    # 1. Initialize result state (safe fallback values)
    state = {
        'ci': (0.5, 0.5, 0.5),
        'dfs': [train_df.copy(), val_df.copy(), test_df.copy()],
        'risks': [np.zeros(len(train_df)), np.zeros(len(val_df)), np.zeros(len(test_df))],
        'best_val_ci': 0.0,
        'patience': 0,
        'metrics_history': []
    }

    # 2. Component initialization
    try:
        model = GNN_Survival(sample_graph, hidden=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=0.001
        )
    except Exception as e:
        print(f"Component initialization failed: {e}")
        return state['ci'], *state['dfs'], *state['risks']

    # 3. Inner training closure
    def run_epoch():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            risk = model(batch)

            if torch.isnan(risk).any(): raise ValueError("NaN risk detected")
            loss = cox_ph_loss(risk, batch.y_time, batch.y_event)
            if torch.isnan(loss): raise ValueError("NaN loss detected")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    # 4. Main training loop
    print(f"\nTraining started: {task_name}")
    try:
        for epoch in range(1, epochs + 1):
            # A. Training step
            loss = run_epoch()
            scheduler.step(loss)

            # B. Evaluation step
            tr_ci, _ = evaluate_Cindex(train_loader, model)
            val_ci, _ = evaluate_Cindex(val_loader, model)
            te_ci, _ = evaluate_Cindex(test_loader, model)

            # Log metrics
            state['metrics_history'].append({'epoch': epoch, 'loss': loss, 'val_ci': val_ci[0]})
            print(
                f'Epoch {epoch:03d} | Loss {loss:.3f} | Train CI {tr_ci[0]:.3f} | Val CI {val_ci[0]:.3f} | Test CI {te_ci[0]:.3f}')

            # C. Validation improvement check
            is_first_valid = (epoch == 1 and not np.isnan(val_ci[0]))
            is_improvement = (val_ci[0] > state['best_val_ci'])

            if is_first_valid or is_improvement:
                state['best_val_ci'] = val_ci[0]
                state['best_epoch'] = epoch
                state['patience'] = 0

                # Save best model weights
                torch.save(model.state_dict(), task_dir / f'best_survival_{task_name}.pt')

                # Capture evaluation data
                _, tr_res, _, _ = evaluate_Cindex(train_loader, model, train_df)
                _, va_res, _, _ = evaluate_Cindex(val_loader, model, val_df)
                _, te_res, _, _ = evaluate_Cindex(test_loader, model, test_df)

                state.update({
                    'tr_ci': tr_ci[0], 'tr_lo': tr_ci[1], 'tr_hi': tr_ci[2],
                    'val_ci': val_ci[0], 'val_lo': val_ci[1], 'val_hi': val_ci[2],
                    'te_ci': te_ci[0], 'te_lo': te_ci[1], 'te_hi': te_ci[2],
                    'train_surv_df': tr_res,
                    'val_surv_df': va_res,
                    'test_surv_df': te_res
                })
                print(
                    f"  >>> {'Baseline' if is_first_valid else 'Improvement'} captured at Epoch {epoch}: Val CI {val_ci[0]:.3f}")
            else:
                state['patience'] += 1
                if state['patience'] >= 5:
                    print("Early stopping triggered")
                    break
    except Exception as e:
        print(f"\nTraining interrupted (task: {task_name}): {e}")
        best_path = task_dir / f'best_survival_{task_name}.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path))

        save_final_results(task_name, state, state['metrics_history'], is_error=True)
    else:
        save_final_results(task_name, state, state['metrics_history'], is_error=False)

    # Update final C-index (fix: state['ci'] was stuck at default 0.5)
    if 'te_ci' in state:
        state['ci'] = (state['te_ci'], state['te_lo'], state['te_hi'])

    # 5. Return (strictly matching expected format)
    return *state['ci'], *state['dfs'], *state['risks']


def save_final_results(task_name, best_results, epoch_metrics, is_error):
    """Save final results: best C-index (with 95% CI) and per-epoch training history."""
    task_dir = epoch_data_dir / task_name
    task_dir.mkdir(exist_ok=True)

    # 1. Save best C-index results (with 95% CI)
    best_ci_results = pd.DataFrame({
        'Dataset': ['train', 'validation', 'test'],
        'Cindex': [
            best_results.get('tr_ci', np.nan),
            best_results.get('val_ci', np.nan),
            best_results.get('te_ci', np.nan)
        ],
        '95%CI_lower': [
            best_results.get('tr_lo', np.nan),
            best_results.get('val_lo', np.nan),
            best_results.get('te_lo', np.nan)
        ],
        '95%CI_upper': [
            best_results.get('tr_hi', np.nan),
            best_results.get('val_hi', np.nan),
            best_results.get('te_hi', np.nan)
        ],
        'Best_Epoch': best_results.get('best_epoch', np.nan)
    })

    status = "error_last_valid" if is_error else "final_best"
    best_ci_results.to_excel(
        task_dir / f"{task_name}_best_cindex_{status}.xlsx",
        index=False
    )

    # 2. Save C-index epoch history
    epoch_metrics_df = pd.DataFrame(epoch_metrics)
    epoch_metrics_df.to_excel(
        task_dir / f"{task_name}_cindex_epoch_history_{status}.xlsx",
        index=False
    )

    # 3. Save survival risk scores (if available)
    if best_results.get('train_surv_df') is not None:
        best_results['train_surv_df'].to_csv(
            task_dir / f"{task_name}_train_survival_risk_{status}.csv",
            index=False
        )
        best_results['val_surv_df'].to_csv(
            task_dir / f"{task_name}_val_survival_risk_{status}.csv",
            index=False
        )
        best_results['test_surv_df'].to_csv(
            task_dir / f"{task_name}_test_survival_risk_{status}.csv",
            index=False
        )

    print(f"Results saved to: {task_dir}")


def get_graph_config(feature_type):
    """
    Return the (node_list, edge_list) for the heterogeneous graph topology
    corresponding to the given feature_type string.
    """
    nodes = []
    edges = []

    def add_bi_edge(src, dst):
        edges.append((src, dst))

    # --- Phase 1: Core layer (f1, t1, f2, t2) ---
    if feature_type.startswith('f1t1'):
        nodes.extend(['f1', 't1'])
        add_bi_edge('f1', 't1')

    if 'f2' in feature_type:
        nodes.append('f2')
        nodes.append('t2')
        add_bi_edge('f1', 'f2')
        add_bi_edge('t1', 'f2')
        add_bi_edge('t1', 't2')
        add_bi_edge('f2', 't2')

    # --- Phase 2: Bridge layer (f3) ---
    if 'f3' in feature_type:
        nodes.append('f3')
        add_bi_edge('f2', 'f3')
        add_bi_edge('t2', 'f3')

    # --- Phase 3: Linear tail (f4 to f9) ---
    tail_match = re.search(r'f3(\d+)', feature_type)
    if tail_match:
        tail_nums = tail_match.group(1)
        current_prev = 'f3'
        for num_char in tail_nums:
            curr_node = f'f{num_char}'
            nodes.append(curr_node)
            add_bi_edge(current_prev, curr_node)
            current_prev = curr_node

    return nodes, edges


def build_hetero_graph(row, nodes, edges, col_map):
    """Build a single-sample heterogeneous graph from one data row."""
    d = HeteroData()

    # 1. Build node features dynamically
    for node_name in nodes:
        if node_name not in col_map:
            raise ValueError(f"Missing node in column map: {node_name}")

        col_var = col_map[node_name]
        vals = row[col_var].values.astype(np.float32)
        d[node_name].x = torch.from_numpy(vals).unsqueeze(0)

    # 2. Build edges (bidirectional)
    ei = torch.tensor([[0], [0]], dtype=torch.long)

    for src, dst in edges:
        d[src, 'to', dst].edge_index = ei
        d[dst, 'to', src].edge_index = ei

    # 3. Set labels
    d.y = torch.tensor(int(row['5year-S']), dtype=torch.long).unsqueeze(0)
    d.y_event = torch.tensor(int(row['is_death']), dtype=torch.float)
    d.y_time = torch.tensor(float(row['duration']), dtype=torch.float)

    return d


def call_build_hetero_graph(train_df, val_df, test_df, feature_type):
    """Entry point for building heterogeneous graph datasets and data loaders."""
    col_map = {
        'f1': f1, 't1': t1,
        'f2': f2, 't2': t2,
        'f3': f3, 'f4': f4, 'f5': f5,
        'f6': f6, 'f7': f7, 'f8': f8, 'f9': f9
    }

    try:
        nodes, edges = get_graph_config(feature_type)
    except Exception as e:
        raise ValueError(f"Failed to parse feature type {feature_type}: {str(e)}")

    print(f"Building graphs for [{feature_type}]")
    print(f"Active Nodes: {nodes}")
    print(f"Active Edges (bidirectional): {edges}")

    build_fn = lambda r: build_hetero_graph(r, nodes, edges, col_map)

    train_graphs = [build_fn(r) for _, r in train_df.iterrows()]
    val_graphs = [build_fn(r) for _, r in val_df.iterrows()]
    test_graphs = [build_fn(r) for _, r in test_df.iterrows()]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_graphs


def run_task(feature_type, data_path, neighbor_vectors, update_features,
             update_model=None,
             prev_train_ids=None, prev_val_ids=None, prev_test_ids=None):
    """
    Execute a single task: feature update + survival model training + evaluation.
    Accepts optional previous split IDs to maintain consistent data splits across tasks.
    """
    print(f"\n{'=' * 20} Task: {feature_type} {'=' * 20}")

    # 1. Pack legacy IDs
    prev_ids = None
    if prev_train_ids is not None:
        prev_ids = {
            'train': prev_train_ids,
            'val': prev_val_ids,
            'test': prev_test_ids
        }

    df = pd.read_excel(data_path)
    df = discrete(df)
    df = select_cols(df)

    # 2. Data splitting
    splits = {}
    if prev_ids is None:
        tr, va, te = data_split(df, '5year-S')
        splits = {'train': tr, 'val': va, 'test': te}
    else:
        for key, saved_ids in prev_ids.items():
            splits[key] = df[df['ID'].isin(saved_ids)].copy()
            if len(splits[key]) < 3:
                raise ValueError(f"Split '{key}' is too small (n={len(splits[key])}). Check ID overlap.")

    # 3. Feature update loop
    updated_splits, scaler = {}, None
    task_is_healthy = True

    if ENABLE_FEATURE_UPDATE:
        for name in ['train', 'val', 'test']:
            is_train = (name == 'train')
            res, _, model_ptr, scaler_ptr, healthy_flag = update_features_with_gnn(
                splits[name], neighbor_vectors, update_features, feature_type,
                model=update_model,
                scaler=scaler if not is_train else None,
                train_model=(update_model is None and is_train),
                dataset_name=name
            )
            updated_splits[name] = res
            if not healthy_flag: task_is_healthy = False
            if is_train: update_model, scaler = model_ptr, scaler_ptr
    else:
        updated_splits = {k: v.copy() for k, v in splits.items()}

    # 4. Survival analysis
    loaders = call_build_hetero_graph(*[updated_splits[k] for k in ['train', 'val', 'test']], feature_type)

    surv_results = train_survival_model(
        *loaders[:3],
        *[updated_splits[k] for k in ['train', 'val', 'test']],
        loaders[3][0],
        feature_type, update_features
    )
    te_ci, te_lo, te_hi, _, _, _, tr_risk, va_risk, te_risk = surv_results

    # 5. Build result summary
    display_name = feature_type if task_is_healthy else f"{feature_type}_best_cindex_error_last_valid"

    summary = {
        "Task": display_name,
        "Survival_Cindex": f"{te_ci:.3f} ({te_lo:.3f}-{te_hi:.3f})",
        **{f"{k}_n": len(v) for k, v in updated_splits.items()}
    }

    current_train_ids = set(updated_splits['train']['ID'])
    current_val_ids = set(updated_splits['val']['ID'])
    current_test_ids = set(updated_splits['test']['ID'])

    return summary, update_model, current_train_ids, current_val_ids, current_test_ids


def main():
    """Main function: iterate through all feature configurations and train survival models."""
    print(f"Device: {device} | Mode: Sequential task execution")

    # 1. Configuration
    f_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    # filenames = [
    #     "f1t1.xlsx", "f1t1f2t2.xlsx", "f1t1f2t2f3.xlsx",
    #     "f1t1f2t2f34.xlsx", "f1t1f2t2f345.xlsx", "f1t1f2t2f3456.xlsx",
    #     "f1t1f2t2f34567.xlsx", "f1t1f2t2f345678.xlsx", "f1t1f2t2f3456789.xlsx"
    # ]
    filenames = ["f1t1f2t2f3456789.xlsx"]

    # 2. Dynamically generate all tasks
    all_tasks = []
    for i in range(len(filenames)):
        current_features = sum(f_list[:i + 1], [])
        all_tasks.append({
            "feature_type": filenames[i].replace(".xlsx", ""),
            "data_path": filenames[i],
            "neighbor_vectors": current_features,
            "update_features": current_features
        })

    # 3. Execute tasks
    tasks = all_tasks

    shared_ids = {'prev_train_ids': None, 'prev_val_ids': None, 'prev_test_ids': None}
    all_results = []

    for i, task in enumerate(tasks):
        print(f"\n{'#' * 30} Running task {i + 1}/{len(tasks)}: {task['feature_type']} {'#' * 30}")

        try:
            # Each task creates its own model (dimensions vary across tasks)
            result, _, *new_ids = run_task(
                **task,
                **shared_ids,
                update_model=None
            )
            all_results.append(result)

            # Lock data splits after the first task
            if i == 0:
                shared_ids = {
                    'prev_train_ids': new_ids[0],
                    'prev_val_ids': new_ids[1],
                    'prev_test_ids': new_ids[2]
                }

        except Exception as e:
            print(f"❌ Task {task['feature_type']} failed: {e}")
            all_results.append({"Task": task["feature_type"], "Status": "Failed", "Error": str(e)})
            continue

    # Save summary
    pd.DataFrame(all_results).to_excel(result_root / "results_summary.xlsx", index=False)


if __name__ == "__main__":
    main()
