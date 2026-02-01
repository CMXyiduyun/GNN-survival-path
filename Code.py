import os
import sys
import random
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from lifelines.utils import concordance_index

# Suppress warnings
warnings.simplefilter('ignore')


# -------------------------------------------------- 1. Configuration --------------------------------------------------
@dataclass
class Config:
    # Paths
    working_dir: str = 'Data/'
    result_root: Path = Path("Results/")

    # Model Params
    seed: int = 20250715
    batch_size: int = 64
    lr_init: float = 0.001
    epochs_gnn: int = 10
    epochs_pred: int = 50
    hidden_dim: int = 32
    max_neighbors: int = 20
    cosine_thres: float = 0.05
    enable_feature_update: bool = True

    # Device
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def __post_init__(self):
        # Create directories
        self.feature_update_dir = self.result_root / "Feature_Update_Results"
        self.prediction_dir = self.result_root / "Survival_Prediction_Results"
        self.feature_update_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True


# Initialize Config
cfg = Config()


# -------------------------------------------------- 2. Data Definitions --------------------------------------------------
class FeatureDefs:
    STATS = ['ID', 'BCLC A', 'BCLC B', 'BCLC C', 'Age', 'Male', 'HBV', 'HCV']
    BASE_FEATURES = ['ViableLesion', 'NewLesion', 'Amount', 'Diameter', 'Metastasis',
                     'LLNM', 'Varicosity', 'VI', 'AFP', 'CPS', 'ALBI']

    # Base treatments for 1st line
    T1 = ['1st_肝切除', '1st_消融', '1st_栓塞', '1st_灌注', '1st_systemic_treatment']
    # Base treatments for subsequent lines
    T2 = ['2nd_肝切除', '2nd_消融', '2nd_栓塞', '2nd_灌注', '2nd_systemic_treatment']

    SURVIVAL_OUTCOME = ['is_death', 'duration']

    @staticmethod
    def get_step_features(prefix: str) -> List[str]:
        """Dynamically generate feature names for a specific step (e.g., '1st_')"""
        # Define the logic for specific features like Amount=2/3 etc.
        # This matches the expanded columns created in DataHandler.discrete
        expanded = [
            f'{prefix}Amount=2/3', f'{prefix}Amount>3',
            f'{prefix}Diameter=50-70', f'{prefix}Diameter>70',
            f'{prefix}Metastasis', f'{prefix}LLNM', f'{prefix}Varicosity', f'{prefix}VI',
            f'{prefix}AFP=400-800', f'{prefix}AFP>800',
            f'{prefix}CPS=B', f'{prefix}CPS=C',
            f'{prefix}ALBI=2', f'{prefix}ALBI=3'
        ]
        if prefix != '1st_':
            expanded = [f'{prefix}ViableLesion', f'{prefix}NewLesion'] + expanded
        return expanded


# -------------------------------------------------- 3. Data Handling --------------------------------------------------
class DataHandler:
    @staticmethod
    def process_duration(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate duration and age if missing."""
        if 'duration' not in data.columns:
            data['first_HCC'] = pd.to_datetime(data['first_HCC'])
            data['last_followup'] = pd.to_datetime(data['last_followup'])
            data['duration'] = (data['last_followup'] - data['first_HCC']).dt.days / 30.5

        if 'Age' not in data.columns:
            data['birth_date'] = pd.to_datetime(data['birth_date'])
            data['Age'] = (data['first_HCC'] - data['birth_date']).dt.days / 365.25
        return data

    @staticmethod
    def discretize_features(data: pd.DataFrame) -> pd.DataFrame:
        """Elegant discretization loop for all time steps."""
        # Global BCLC
        for stage in ['A', 'B', 'C']:
            val = 'Z' if stage == 'A' else stage  # Handle logic for 'Z'/'A' mapping to A
            if stage == 'A':
                data['BCLC A'] = data['BCLC'].apply(lambda x: 1 if x in ['Z', 'A'] else 0)
            else:
                data[f'BCLC {stage}'] = data['BCLC'].apply(lambda x: 1 if x == stage else 0)

        # Dynamic loop for 1st to 9th
        prefixes = ['1st_', '2nd_', '3rd_', '4th_', '5th_', '6th_', '7th_', '8th_', '9th_']

        for prefix in prefixes:
            # Check if this step exists in data (e.g., by checking a core column like Amount)
            col_amount = f"{prefix}Amount"
            if col_amount not in data.columns:
                continue

            # Amount
            data[f'{prefix}Amount=2/3'] = data[col_amount].apply(lambda x: 1 if 1 < x <= 3 else 0)
            data[f'{prefix}Amount>3'] = data[col_amount].apply(lambda x: 1 if x > 3 else 0)

            # Diameter
            col_dia = f"{prefix}Diameter"
            data[f'{prefix}Diameter=50-70'] = data[col_dia].apply(lambda x: 1 if 50 <= x <= 70 else 0)
            data[f'{prefix}Diameter>70'] = data[col_dia].apply(lambda x: 1 if x > 70 else 0)

            # AFP
            col_afp = f"{prefix}AFP"
            data[f'{prefix}AFP=400-800'] = data[col_afp].apply(lambda x: 1 if 400 <= x <= 800 else 0)
            data[f'{prefix}AFP>800'] = data[col_afp].apply(lambda x: 1 if x > 800 else 0)

            # CPS
            col_cps = f"{prefix}CPS"
            data[f'{prefix}CPS=B'] = data[col_cps].apply(lambda x: 1 if 6 < x <= 9 else 0)
            data[f'{prefix}CPS=C'] = data[col_cps].apply(lambda x: 1 if x >= 10 else 0)

            # ALBI
            col_albi = f"{prefix}ALBI"
            data[f'{prefix}ALBI=2'] = data[col_albi].apply(lambda x: 1 if x == 2 else 0)
            data[f'{prefix}ALBI=3'] = data[col_albi].apply(lambda x: 1 if x == 3 else 0)

            # Systemic Treatment Combination (if columns exist)
            target_col = f"{prefix}targted_drug"
            immuno_col = f"{prefix}immunodrug"
            if target_col in data.columns and immuno_col in data.columns:
                data[f'{prefix}systemic_treatment'] = data[[target_col, immuno_col]].max(axis=1)

        return data

    @staticmethod
    def split_data(df: pd.DataFrame, label_col: str = 'is_death') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into Train (68%), Val (12%), Test (20%)."""
        y = df[label_col].values

        # 1. Split Test (20%)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=cfg.seed)
        trainval_idx, test_idx = next(sss1.split(df, y))
        trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # 2. Split Val (15% of remaining 80% ≈ 12% total)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg.seed)
        train_idx, val_idx = next(sss2.split(trainval_df, trainval_df[label_col]))
        train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df


# -------------------------------------------------- 4. Models (GNN & Prediction) --------------------------------------------------

def cox_ph_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Standard Cox Partial Likelihood Loss."""
    # Sort by duration descending
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    exp_risk = torch.exp(risk_scores)
    cum_exp_risk = torch.cumsum(exp_risk, dim=0)
    log_cum_risk = torch.log(cum_exp_risk)

    # Loss only calculated for censored events
    loss = -torch.sum((risk_scores - log_cum_risk) * events) / (torch.sum(events) + 1e-8)
    return loss


class FeatureRefinementGNN(nn.Module):
    """GNN to refine features based on patient similarity graphs."""

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, in_channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index):
        x_residual = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # Reconstruct features
        x_updated = self.decoder(x)

        # Residual connection: alpha * new + (1-alpha) * old
        return self.alpha * x_updated + (1 - self.alpha) * x_residual


class SurvivalPredictor(nn.Module):
    """DeepSurv-like MLP for risk prediction."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Single risk score output
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------- 5. Core Engine --------------------------------------------------

class GNNGraphBuilder:
    def __init__(self):
        self.neighbor_cache = {}

    def build(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Data, StandardScaler]:
        """Builds a graph where edges connect similar patients."""
        X = df[feature_cols].values

        # Standardize for distance calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute Cosine Distance
        dist_matrix = cosine_distances(X_scaled)
        np.fill_diagonal(dist_matrix, 2.0)  # Ignore self

        edges = set()
        n_samples = len(df)

        # Create edges
        for i in range(n_samples):
            # Find neighbors within threshold
            candidates = np.where(dist_matrix[i] < cfg.cosine_thres)[0]
            # Sort by similarity (distance)
            sorted_idx = np.argsort(dist_matrix[i][candidates])
            closest = candidates[sorted_idx][:cfg.max_neighbors]

            for j in closest:
                if i < j:  # Undirected
                    edges.add((i, j))
                    edges.add((j, i))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
                                                                                                            dtype=torch.long)

        data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),  # Use scaled data for GNN input
            edge_index=edge_index,
            y_event=torch.tensor(df['is_death'].values, dtype=torch.float),
            y_duration=torch.tensor(df['duration'].values, dtype=torch.float)
        )

        return data, scaler


class PipelineEngine:
    def __init__(self):
        self.graph_builder = GNNGraphBuilder()

    def run_gnn_update(self, df: pd.DataFrame, features: List[str], dataset_type: str,
                       model: Optional[nn.Module] = None):
        """Runs the GNN feature update step."""
        if not cfg.enable_feature_update:
            return df, None

        data, scaler = self.graph_builder.build(df, features)
        data = data.to(cfg.device)

        # Initialize or use existing model
        if model is None:
            model = FeatureRefinementGNN(len(features), cfg.hidden_dim).to(cfg.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_init)

            # Train GNN (Self-supervised using Cox Loss on risk features)
            model.train()
            for _ in range(cfg.epochs_gnn):
                optimizer.zero_grad()
                updated_feats = model(data.x, data.edge_index)
                # Simple proxy loss: minimize cox loss of the updated features averaged
                risk = torch.mean(updated_feats, dim=1)
                loss = cox_ph_loss(risk, data.y_duration, data.y_event)
                loss.backward()
                optimizer.step()

        # Inference
        model.eval()
        with torch.no_grad():
            updated_x_scaled = model(data.x, data.edge_index).cpu().numpy()

        # Inverse transform to get original scale back
        updated_x = scaler.inverse_transform(updated_x_scaled)

        # Create new DataFrame
        new_df = df.copy()
        new_df[features] = updated_x

        return new_df, model

    def run_survival_prediction(self, train_df, val_df, test_df, features, task_name):
        """Runs the downstream survival prediction task."""

        def get_tensors(d):
            return (torch.tensor(d[features].values, dtype=torch.float).to(cfg.device),
                    torch.tensor(d['duration'].values, dtype=torch.float).to(cfg.device),
                    torch.tensor(d['is_death'].values, dtype=torch.float).to(cfg.device))

        X_tr, T_tr, E_tr = get_tensors(train_df)
        X_val, T_val, E_val = get_tensors(val_df)
        X_te, T_te, E_te = get_tensors(test_df)

        model = SurvivalPredictor(len(features)).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_init)

        # Training
        best_c_index = 0
        best_state = None

        for epoch in range(cfg.epochs_pred):
            model.train()
            opt.zero_grad()
            risk = model(X_tr)
            loss = cox_ph_loss(risk, T_tr, E_tr)
            loss.backward()
            opt.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_risk = model(X_val)
                try:
                    val_c = concordance_index(T_val.cpu(), -val_risk.cpu(), E_val.cpu())
                except:
                    val_c = 0.5

                if val_c > best_c_index:
                    best_c_index = val_c
                    best_state = model.state_dict()

        # Testing
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            test_risk = model(X_te)
            c_index = concordance_index(T_te.cpu(), -test_risk.cpu(), E_te.cpu())

        print(f"Task [{task_name}] Completed. Test C-Index: {c_index:.4f}")
        return c_index


# -------------------------------------------------- 6. Main Execution --------------------------------------------------

def main():
    if not os.path.exists(cfg.working_dir):
        print("Working directory not found. Please check paths.")
        return

    os.chdir(cfg.working_dir)
    # Load Data (Replace with your actual file name)
    try:
        raw_df = pd.read_excel("f1t1f2t220250715.xlsx")
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Preprocess
    print("Preprocessing data...")
    df = DataHandler.process_duration(raw_df)
    df = DataHandler.discretize_features(df)

    # Define tasks (Features required for each phase)
    # This logic replicates the expanding window of features (Base -> Base+T1 -> Base+T1+F2...)
    tasks = []

    # Base Features (F1)
    f1_cols = FeatureDefs.get_step_features('1st_')
    tasks.append(("Phase1_Base", FeatureDefs.STATS + f1_cols))

    # Treatment 1 (F1 + T1)
    tasks.append(("Phase2_Treatment1", FeatureDefs.STATS + f1_cols + FeatureDefs.T1))

    # Follow-up (F1 + T1 + F2)
    f2_cols = FeatureDefs.get_step_features('2nd_')
    tasks.append(("Phase3_FollowUp1", FeatureDefs.STATS + f1_cols + FeatureDefs.T1 + f2_cols))

    # Engine
    engine = PipelineEngine()
    results = []

    for task_name, feat_cols in tasks:
        # Check availability
        valid_cols = [c for c in feat_cols if c in df.columns]
        curr_df = df[FeatureDefs.SURVIVAL_OUTCOME + valid_cols].dropna()

        # Split
        train, val, test = DataHandler.split_data(curr_df)

        # Update Features (GNN)
        gnn_cols = [c for c in valid_cols if c not in FeatureDefs.STATS]  # Only update numerical/categorical features

        print(f"\n--- Running Task: {task_name} ---")
        train_upd, model = engine.run_gnn_update(train, gnn_cols, "train")
        val_upd, _ = engine.run_gnn_update(val, gnn_cols, "val", model)
        test_upd, _ = engine.run_gnn_update(test, gnn_cols, "test", model)

        # Save Updated Features
        save_path = cfg.feature_update_dir / f"{task_name}_test_features.xlsx"
        test_upd.to_excel(save_path, index=False)

        # Predict Survival
        c_idx = engine.run_survival_prediction(train_upd, val_upd, test_upd, gnn_cols, task_name)

        results.append({'Task': task_name, 'C-Index': c_idx})

    # Save Final Results
    res_df = pd.DataFrame(results)
    res_df.to_excel(cfg.result_root / "Final_Survival_Metrics.xlsx", index=False)
    print("\nPipeline Finished Successfully.")


if __name__ == "__main__":
    main()
