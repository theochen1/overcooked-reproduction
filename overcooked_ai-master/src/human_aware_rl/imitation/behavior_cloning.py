"""
PyTorch Behavior Cloning for Overcooked AI

This module provides behavior cloning (imitation learning) functionality
using PyTorch, replacing the deprecated TensorFlow implementation.
"""

import copy
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.static import CLEAN_2019_HUMAN_DATA_TRAIN
from human_aware_rl.utils import get_flattened_keys, recursive_dict_update
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import DEFAULT_ENV_PARAMS
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


#################
# Configuration #
#################

BC_SAVE_DIR = os.path.join(DATA_DIR, "bc_runs")

DEFAULT_DATA_PARAMS = {
    "layouts": ["cramped_room"],
    "check_trajectories": False,
    "featurize_states": True,
    "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
}

DEFAULT_MLP_PARAMS = {
    # Number of fully connected layers to use in our network
    "num_layers": 2,
    # Each int represents a layer of that hidden size
    "net_arch": [64, 64],
}

DEFAULT_TRAINING_PARAMS = {
    "epochs": 100,
    "validation_split": 0.15,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "adam_epsilon": 1e-8,  # Paper Table 1: explicit for reproducibility
    "use_class_weights": False,
    "patience": 20,  # Early stopping patience
    "lr_patience": 3,  # LR scheduler patience
    "lr_factor": 0.1,  # LR reduction factor
}

DEFAULT_EVALUATION_PARAMS = {
    "ep_length": 400,
    "num_games": 1,
    "display": False,
}

DEFAULT_BC_PARAMS = {
    "use_lstm": False,
    "cell_size": 256,
    "data_params": DEFAULT_DATA_PARAMS,
    "mdp_params": {"layout_name": "cramped_room", "old_dynamics": False},
    "env_params": DEFAULT_ENV_PARAMS,
    "mdp_fn_params": {},
    "mlp_params": DEFAULT_MLP_PARAMS,
    "training_params": DEFAULT_TRAINING_PARAMS,
    "evaluation_params": DEFAULT_EVALUATION_PARAMS,
    "action_shape": (len(Action.ALL_ACTIONS),),
}

# Boolean indicating whether all param dependencies have been loaded
_params_initialized = False


def _get_base_ae(bc_params: Dict) -> AgentEvaluator:
    """Get the base AgentEvaluator from bc_params."""
    return AgentEvaluator.from_layout_name(
        mdp_params=bc_params["mdp_params"],
        env_params=bc_params["env_params"]
    )


def _get_observation_shape(bc_params: Dict) -> Tuple[int, ...]:
    """
    Helper function for creating a dummy environment from "mdp_params" and "env_params" specified
    in bc_params and returning the shape of the observation space
    """
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    dummy_state = base_env.mdp.get_standard_start_state()
    obs_shape = base_env.featurize_state_mdp(dummy_state)[0].shape
    return obs_shape


def get_bc_params(**args_to_override) -> Dict:
    """
    Loads default bc params defined globally. For each key in args_to_override, overrides the default with the
    value specified for that key. Recursively checks all children. If key not found, creates new top level parameter.

    Note: Even though children can share keys, for simplicity, we enforce the condition that all keys at all levels must be distinct
    """
    global _params_initialized, DEFAULT_BC_PARAMS
    if not _params_initialized:
        DEFAULT_BC_PARAMS["observation_shape"] = _get_observation_shape(DEFAULT_BC_PARAMS)
        _params_initialized = True
    params = copy.deepcopy(DEFAULT_BC_PARAMS)

    for arg, val in args_to_override.items():
        updated = recursive_dict_update(params, arg, val)
        if not updated:
            print(f"WARNING, no value for specified bc argument {arg} found in schema. Adding as top level parameter")
            params[arg] = val

    all_keys = get_flattened_keys(params)
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Every key at every level must be distinct for BC params!")

    return params


##############
# Model code #
##############


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the last axis."""
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class BCModel(nn.Module):
    """
    MLP-based Behavior Cloning model for Overcooked.
    Maps featurized observations to action logits.
    """

    def __init__(self, observation_shape: Tuple[int, ...], action_shape: Tuple[int, ...], mlp_params: Dict, **kwargs):
        super(BCModel, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        input_dim = int(np.prod(observation_shape))
        output_dim = action_shape[0]
        
        # Build fully connected layers
        assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"
        
        layers = []
        prev_dim = input_dim
        for i in range(mlp_params["num_layers"]):
            units = mlp_params["net_arch"][i]
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, *observation_shape)
            
        Returns:
            Action logits of shape (batch_size, num_actions)
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class BCLSTMModel(nn.Module):
    """
    LSTM-based Behavior Cloning model for Overcooked.
    Maps sequences of featurized observations to action logits.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        mlp_params: Dict,
        cell_size: int = 256,
        **kwargs
    ):
        super(BCLSTMModel, self).__init__()
        
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.cell_size = cell_size
        
        input_dim = int(np.prod(observation_shape))
        output_dim = action_shape[0]
        
        # Build fully connected layers before LSTM
        assert len(mlp_params["net_arch"]) == mlp_params["num_layers"], "Invalid Fully Connected params"
        
        fc_layers = []
        prev_dim = input_dim
        for i in range(mlp_params["num_layers"]):
            units = mlp_params["net_arch"][i]
            fc_layers.append(nn.Linear(prev_dim, units))
            fc_layers.append(nn.ReLU())
            prev_dim = units
        
        self.fc_network = nn.Sequential(*fc_layers)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=prev_dim,
            hidden_size=cell_size,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(cell_size, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, *observation_shape)
            hidden: Optional tuple of (h_0, c_0) each of shape (1, batch_size, cell_size)
            seq_lens: Optional sequence lengths for masking
            
        Returns:
            Tuple of (logits, (h_n, c_n))
            - logits: shape (batch_size, seq_len, num_actions)
            - h_n, c_n: shape (1, batch_size, cell_size)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Flatten the observation dimensions
        x = x.view(batch_size * seq_len, -1)
        
        # Apply FC layers
        x = self.fc_network(x)
        
        # Reshape back to sequence
        x = x.view(batch_size, seq_len, -1)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h_0 = torch.zeros(1, batch_size, self.cell_size, device=x.device)
            c_0 = torch.zeros(1, batch_size, self.cell_size, device=x.device)
            hidden = (h_0, c_0)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply output layer to all timesteps
        logits = self.output_layer(lstm_out)
        
        return logits, (h_n, c_n)

    def get_initial_state(self, batch_size: int = 1, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Return initial hidden state for LSTM."""
        h_0 = torch.zeros(1, batch_size, self.cell_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.cell_size, device=device)
        return h_0, c_0


def build_bc_model(use_lstm: bool = False, **kwargs) -> nn.Module:
    """
    Build and return a BC model based on parameters.
    
    Args:
        use_lstm: Whether to use LSTM model
        **kwargs: Parameters including observation_shape, action_shape, mlp_params, cell_size
        
    Returns:
        PyTorch model (BCModel or BCLSTMModel)
    """
    if use_lstm:
        return BCLSTMModel(**kwargs)
    else:
        return BCModel(**kwargs)


#################
# Data loading #
#################


def _pad_sequences(sequences: List[np.ndarray], max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of arrays, each of shape (seq_len, ...)
        max_len: Maximum sequence length (computed if None)
        
    Returns:
        Tuple of (padded_sequences, sequence_lengths)
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    seq_lens = np.array([len(seq) for seq in sequences])
    
    # Get shape of each element
    elem_shape = sequences[0][0].shape if hasattr(sequences[0][0], 'shape') else ()
    
    # Create padded array
    padded = np.zeros((len(sequences), max_len, *elem_shape), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = np.array(seq)
    
    return padded, seq_lens


def load_data(bc_params: Dict, verbose: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Load and preprocess human demonstration data.
    
    Args:
        bc_params: BC parameters including data_params
        verbose: Whether to print loading info
        
    Returns:
        Tuple of (inputs, seq_lens, targets)
        - For MLP: inputs (N, obs_dim), seq_lens=None, targets (N, 1)
        - For LSTM: inputs (N, max_seq_len, obs_dim), seq_lens (N,), targets (N, max_seq_len, 1)
    """
    processed_trajs = get_human_human_trajectories(
        **bc_params["data_params"], silent=not verbose
    )
    inputs = processed_trajs["ep_states"]
    targets = processed_trajs["ep_actions"]

    if bc_params["use_lstm"]:
        # Pad sequences
        inputs_padded, seq_lens = _pad_sequences(inputs)
        targets_padded, _ = _pad_sequences(targets)
        return inputs_padded, seq_lens, targets_padded
    else:
        # Flatten all episodes
        inputs_flat = np.vstack(inputs).astype(np.float32)
        targets_flat = np.vstack(targets).astype(np.int64).flatten()
        return inputs_flat, None, targets_flat


#################
# Training     #
#################


class EarlyStopping:
    """Early stopping to stop training when loss doesn't improve."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        return self.should_stop


def train_bc_model(
    model_dir: str,
    bc_params: Dict,
    verbose: bool = False,
    device: Optional[str] = None
) -> nn.Module:
    """
    Train a behavior cloning model.
    
    Args:
        model_dir: Directory to save model and logs
        bc_params: BC parameters
        verbose: Whether to print training progress
        device: Device to train on ('cpu', 'cuda', or None for auto)
        
    Returns:
        Trained PyTorch model
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    
    # Load data
    inputs, seq_lens, targets = load_data(bc_params, verbose)
    
    training_params = bc_params["training_params"]
    
    # Compute class weights if needed
    if training_params["use_class_weights"]:
        if bc_params["use_lstm"]:
            flat_targets = targets.flatten()
        else:
            flat_targets = targets
        classes, counts = np.unique(flat_targets[flat_targets >= 0], return_counts=True)
        weights = torch.tensor(np.sum(counts) / counts, dtype=torch.float32, device=device)
    else:
        weights = None
    
    # Create model
    model = build_bc_model(**bc_params)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params["learning_rate"],
        eps=training_params.get("adam_epsilon", 1e-8),  # Paper Table 1
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=training_params["lr_factor"],
        patience=training_params["lr_patience"]
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=training_params["patience"])
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "logs"))
    
    # Split data into train/val
    val_split = training_params["validation_split"]
    n_samples = len(inputs)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    if bc_params["use_lstm"]:
        train_inputs = torch.tensor(inputs[train_idx], dtype=torch.float32)
        train_targets = torch.tensor(targets[train_idx], dtype=torch.long)
        train_seq_lens = torch.tensor(seq_lens[train_idx], dtype=torch.long)
        val_inputs = torch.tensor(inputs[val_idx], dtype=torch.float32)
        val_targets = torch.tensor(targets[val_idx], dtype=torch.long)
        val_seq_lens = torch.tensor(seq_lens[val_idx], dtype=torch.long)
        
        # For LSTM, batch size is 1 for simplicity
        batch_size = 1
        train_dataset = TensorDataset(train_inputs, train_targets, train_seq_lens)
        val_dataset = TensorDataset(val_inputs, val_targets, val_seq_lens)
    else:
        train_inputs = torch.tensor(inputs[train_idx], dtype=torch.float32)
        train_targets = torch.tensor(targets[train_idx], dtype=torch.long)
        val_inputs = torch.tensor(inputs[val_idx], dtype=torch.float32)
        val_targets = torch.tensor(targets[val_idx], dtype=torch.long)
        
        batch_size = training_params["batch_size"]
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(training_params["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if bc_params["use_lstm"]:
                batch_inputs, batch_targets, batch_seq_lens = batch
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(batch_inputs)
                
                # Compute loss only for valid timesteps
                loss = 0.0
                for i, seq_len in enumerate(batch_seq_lens):
                    seq_logits = logits[i, :seq_len].reshape(-1, logits.shape[-1])
                    seq_targets = batch_targets[i, :seq_len].reshape(-1)
                    if weights is not None:
                        loss += F.cross_entropy(seq_logits, seq_targets, weight=weights)
                    else:
                        loss += F.cross_entropy(seq_logits, seq_targets)
                    
                    preds = seq_logits.argmax(dim=-1)
                    train_correct += (preds == seq_targets).sum().item()
                    train_total += seq_len.item()
                
                loss = loss / len(batch_seq_lens)
            else:
                batch_inputs, batch_targets = batch
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_inputs)
                
                if weights is not None:
                    loss = F.cross_entropy(logits, batch_targets, weight=weights)
                else:
                    loss = F.cross_entropy(logits, batch_targets)
                
                preds = logits.argmax(dim=-1)
                train_correct += (preds == batch_targets).sum().item()
                train_total += len(batch_targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if bc_params["use_lstm"]:
                    batch_inputs, batch_targets, batch_seq_lens = batch
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    logits, _ = model(batch_inputs)
                    
                    loss = 0.0
                    for i, seq_len in enumerate(batch_seq_lens):
                        seq_logits = logits[i, :seq_len].reshape(-1, logits.shape[-1])
                        seq_targets = batch_targets[i, :seq_len].reshape(-1)
                        loss += F.cross_entropy(seq_logits, seq_targets)
                        
                        preds = seq_logits.argmax(dim=-1)
                        val_correct += (preds == seq_targets).sum().item()
                        val_total += seq_len.item()
                    
                    loss = loss / len(batch_seq_lens)
                else:
                    batch_inputs, batch_targets = batch
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    logits = model(batch_inputs)
                    loss = F.cross_entropy(logits, batch_targets)
                    
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == batch_targets).sum().item()
                    val_total += len(batch_targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        
        if verbose:
            print(f"Epoch {epoch+1}/{training_params['epochs']}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_bc_model(model_dir, model, bc_params, verbose=False)
        
        # Early stopping
        if early_stopping(val_loss):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    
    # Load best model
    model, _ = load_bc_model(model_dir, verbose=False)
    model = model.to(device)
    
    return model


def save_bc_model(model_dir: str, model: nn.Module, bc_params: Dict, verbose: bool = False) -> None:
    """
    Save a BC model to disk.
    
    Args:
        model_dir: Directory to save model
        model: PyTorch model to save
        bc_params: BC parameters
        verbose: Whether to print save info
    """
    if verbose:
        print(f"Saving bc model at {model_dir}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(model_dir, "model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_class": type(model).__name__,
    }, model_path)
    
    # Save metadata
    metadata_path = os.path.join(model_dir, "metadata.pickle")
    with open(metadata_path, "wb") as f:
        pickle.dump(bc_params, f)


def load_bc_model(model_dir: str, verbose: bool = False, device: Optional[str] = None) -> Tuple[nn.Module, Dict]:
    """
    Load a BC model from disk.
    
    Args:
        model_dir: Directory containing saved model
        verbose: Whether to print load info
        device: Device to load model to
        
    Returns:
        Tuple of (model, bc_params)
    """
    if verbose:
        print(f"Loading bc model from {model_dir}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.pickle")
    with open(metadata_path, "rb") as f:
        bc_params = pickle.load(f)
    
    # Create model
    model = build_bc_model(**bc_params)
    
    # Load model weights
    model_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, bc_params


def evaluate_bc_model(model: nn.Module, bc_params: Dict, verbose: bool = False) -> float:
    """
    Evaluate a BC model by running rollouts in the Overcooked environment.
    
    Args:
        model: Trained BC model
        bc_params: BC parameters
        verbose: Whether to print evaluation info
        
    Returns:
        Average sparse reward achieved during evaluation
    """
    from human_aware_rl.imitation.bc_agent import BCAgent
    from overcooked_ai_py.agents.agent import AgentPair
    
    evaluation_params = bc_params["evaluation_params"]
    
    # Get agent evaluator
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    
    def featurize_fn(state):
        return base_env.featurize_state_mdp(state)
    
    # Create BC agents
    agent_0 = BCAgent(model, bc_params, featurize_fn, agent_index=0, stochastic=True)
    agent_1 = BCAgent(model, bc_params, featurize_fn, agent_index=1, stochastic=True)
    agent_pair = AgentPair(agent_0, agent_1)
    
    # Run evaluation
    results = base_ae.evaluate_agent_pair(
        agent_pair,
        num_games=evaluation_params["num_games"],
        display=evaluation_params["display"],
        info=verbose
    )
    
    # Compute average sparse return
    reward = np.mean(results["ep_returns"])
    return reward


if __name__ == "__main__":
    params = get_bc_params()
    model = train_bc_model(os.path.join(BC_SAVE_DIR, "default"), params, verbose=True)
    # Evaluate our model's performance in a rollout
    reward = evaluate_bc_model(model, params)
    print(f"Evaluation reward: {reward}")

