"""
Unit tests for PyTorch Behavior Cloning implementation.

These tests verify:
- Model construction (MLP and LSTM)
- Model save/load functionality
- Training loop
- Agent evaluation
"""

import gc
import os
import pickle
import random
import shutil
import time
import unittest
import warnings
from typing import Dict

import numpy as np
import torch

from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.imitation.behavior_cloning import (
    BC_SAVE_DIR,
    build_bc_model,
    evaluate_bc_model,
    get_bc_params,
    load_bc_model,
    save_bc_model,
    train_bc_model,
)
from human_aware_rl.static import (
    BC_EXPECTED_DATA_PATH,
    DUMMY_2019_CLEAN_HUMAN_DATA_PATH,
)


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_pickle():
    """Clear the expected results pickle file."""
    with open(BC_EXPECTED_DATA_PATH, "wb") as f:
        pickle.dump({}, f)


class TestBCTraining(unittest.TestCase):
    """
    Unit tests for behavior cloning training and utilities.

    Attributes:
        compute_pickle (bool): Whether to store results as expected values for future tests
        strict (bool): Whether to compare against expected values for exact match
        min_performance (int): Minimum reward achieved in BC-BC rollout after training
    """

    def __init__(self, test_name):
        super(TestBCTraining, self).__init__(test_name)
        self.compute_pickle = False
        self.strict = False
        self.min_performance = 0
        assert not (self.compute_pickle and self.strict), (
            "Cannot compute pickle and run strict reproducibility tests at same time"
        )
        if self.compute_pickle:
            _clear_pickle()

    def setUp(self):
        set_global_seed(0)
        print(f"\nIn Class {self.__class__.__name__}, in Method {self._testMethodName}")
        
        # Suppress warnings
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        
        # Setup BC params
        self.bc_params = get_bc_params(
            **{"data_path": DUMMY_2019_CLEAN_HUMAN_DATA_PATH}
        )
        self.bc_params["mdp_params"]["layout_name"] = "cramped_room"
        self.bc_params["training_params"]["epochs"] = 1
        self.model_dir = os.path.join(BC_SAVE_DIR, "test_model")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Load dummy input for testing
        processed_trajs, _ = get_trajs_from_data(
            **self.bc_params["data_params"], silent=True
        )
        self.dummy_input = np.vstack(processed_trajs["ep_states"])[:1, :].astype(np.float32)
        self.initial_states = (
            torch.zeros(1, 1, self.bc_params["cell_size"]),
            torch.zeros(1, 1, self.bc_params["cell_size"]),
        )
        
        # Load expected values
        # Note: Old pickle files may contain TensorFlow objects which will fail to load
        try:
            with open(BC_EXPECTED_DATA_PATH, "rb") as f:
                self.expected = pickle.load(f)
        except (FileNotFoundError, EOFError, ModuleNotFoundError, ImportError):
            self.expected = {}

    def tearDown(self):
        if self.compute_pickle:
            with open(BC_EXPECTED_DATA_PATH, "wb") as f:
                pickle.dump(self.expected, f)

        # Force garbage collection
        gc.collect()
        time.sleep(0.1)

        try:
            shutil.rmtree(self.model_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not fully remove directory {self.model_dir}: {e}")

    def _forward_model(self, model, obs):
        """Helper to run forward pass on model."""
        model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            output = model(obs_tensor)
            return output.numpy()

    def test_model_construction(self):
        """Test that model can be constructed with correct output shape."""
        model = build_bc_model(**self.bc_params)
        
        # Check output shape
        output = self._forward_model(model, self.dummy_input)
        self.assertEqual(output.shape, (1, self.bc_params["action_shape"][0]))
        
        if self.compute_pickle:
            self.expected["test_model_construction"] = output
        if self.strict:
            self.assertTrue(
                np.allclose(output, self.expected["test_model_construction"])
            )

    def test_save_and_load(self):
        """Test model save and load functionality."""
        model = build_bc_model(**self.bc_params)
        
        # Get output before saving
        original_output = self._forward_model(model, self.dummy_input)
        
        # Save and load
        save_bc_model(self.model_dir, model, self.bc_params)
        loaded_model, loaded_params = load_bc_model(self.model_dir)
        
        # Check params match
        # Note: We compare essential keys since loaded params might have minor differences
        for key in ["use_lstm", "cell_size", "observation_shape", "action_shape"]:
            if key in self.bc_params and key in loaded_params:
                self.assertEqual(self.bc_params[key], loaded_params[key])
        
        # Check outputs match
        loaded_output = self._forward_model(loaded_model, self.dummy_input)
        self.assertTrue(np.allclose(original_output, loaded_output, rtol=1e-5))

    def test_training(self):
        """Test that model can be trained without errors."""
        model = train_bc_model(self.model_dir, self.bc_params)
        
        # Check model produces valid output
        output = self._forward_model(model, self.dummy_input)
        self.assertEqual(output.shape, (1, self.bc_params["action_shape"][0]))
        
        # Check that model files were saved
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "model.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.model_dir, "metadata.pickle")))
        
        if self.compute_pickle:
            self.expected["test_training"] = output
        if self.strict:
            self.assertTrue(
                np.allclose(output, self.expected["test_training"])
            )

    def test_agent_evaluation(self):
        """Test that trained BC agents can be evaluated in the environment."""
        self.bc_params["training_params"]["epochs"] = 20
        model = train_bc_model(self.model_dir, self.bc_params, verbose=False)
        results = evaluate_bc_model(model, self.bc_params, verbose=False)

        # Sanity check - should achieve at least minimum performance
        self.assertGreaterEqual(results, self.min_performance)

        if self.compute_pickle:
            self.expected["test_agent_evaluation"] = results
        if self.strict:
            self.assertAlmostEqual(results, self.expected["test_agent_evaluation"])


class TestBCTrainingLSTM(TestBCTraining):
    """Unit tests for LSTM-based behavior cloning."""

    def _forward_lstm(self, model, obs, states=None):
        """Helper to run forward pass on LSTM model."""
        model.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            # Add batch and sequence dimensions
            obs_tensor = obs_tensor.unsqueeze(1)  # (batch, 1, obs_dim)
            
            if states is None:
                states = self.initial_states
            
            logits, (h_out, c_out) = model(obs_tensor, states)
            logits = logits.squeeze(1)  # (batch, num_actions)
            return logits.numpy(), (h_out, c_out)

    def test_lstm_construction(self):
        """Test that LSTM model can be constructed with correct output shape."""
        self.bc_params["use_lstm"] = True
        model = build_bc_model(**self.bc_params)
        
        # Check output shape
        output, states = self._forward_lstm(model, self.dummy_input)
        self.assertEqual(output.shape, (1, self.bc_params["action_shape"][0]))
        self.assertEqual(states[0].shape, (1, 1, self.bc_params["cell_size"]))
        self.assertEqual(states[1].shape, (1, 1, self.bc_params["cell_size"]))
        
        if self.compute_pickle:
            self.expected["test_lstm_construction"] = output
        if self.strict:
            self.assertTrue(
                np.allclose(output, self.expected["test_lstm_construction"])
            )

    def test_lstm_training(self):
        """Test that LSTM model can be trained without errors."""
        self.bc_params["use_lstm"] = True
        model = train_bc_model(self.model_dir, self.bc_params)
        
        # Check model produces valid output
        output, _ = self._forward_lstm(model, self.dummy_input)
        self.assertEqual(output.shape, (1, self.bc_params["action_shape"][0]))
        
        if self.compute_pickle:
            self.expected["test_lstm_training"] = output
        if self.strict:
            self.assertTrue(
                np.allclose(output, self.expected["test_lstm_training"])
            )

    def test_lstm_evaluation(self):
        """Test that trained LSTM BC agents can be evaluated in the environment."""
        self.bc_params["use_lstm"] = True
        self.bc_params["training_params"]["epochs"] = 1
        model = train_bc_model(self.model_dir, self.bc_params, verbose=False)
        results = evaluate_bc_model(model, self.bc_params, verbose=False)

        # Sanity check
        self.assertGreaterEqual(results, self.min_performance)

        if self.compute_pickle:
            self.expected["test_lstm_evaluation"] = results
        if self.strict:
            self.assertAlmostEqual(results, self.expected["test_lstm_evaluation"])

    def test_lstm_save_and_load(self):
        """Test LSTM model save and load functionality."""
        self.bc_params["use_lstm"] = True
        model = build_bc_model(**self.bc_params)
        
        # Get output before saving
        original_output, _ = self._forward_lstm(model, self.dummy_input)
        
        # Save and load
        save_bc_model(self.model_dir, model, self.bc_params)
        loaded_model, loaded_params = load_bc_model(self.model_dir)
        
        # Check outputs match
        loaded_output, _ = self._forward_lstm(loaded_model, self.dummy_input)
        self.assertTrue(np.allclose(original_output, loaded_output, rtol=1e-5))


class TestBCAgent(unittest.TestCase):
    """Unit tests for BCAgent class."""

    def setUp(self):
        set_global_seed(0)
        warnings.simplefilter("ignore", ResourceWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        
        self.bc_params = get_bc_params(
            **{"data_path": DUMMY_2019_CLEAN_HUMAN_DATA_PATH}
        )
        self.bc_params["mdp_params"]["layout_name"] = "cramped_room"
        self.model_dir = os.path.join(BC_SAVE_DIR, "test_agent_model")
        
        os.makedirs(self.model_dir, exist_ok=True)

    def tearDown(self):
        gc.collect()
        time.sleep(0.1)
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def test_bc_agent_action(self):
        """Test that BCAgent can produce valid actions."""
        from human_aware_rl.imitation.bc_agent import BCAgent
        from overcooked_ai_py.agents.benchmarking import AgentEvaluator
        from overcooked_ai_py.mdp.actions import Action
        
        # Build and save a model
        model = build_bc_model(**self.bc_params)
        save_bc_model(self.model_dir, model, self.bc_params)
        
        # Load model and create agent
        model, bc_params = load_bc_model(self.model_dir)
        
        # Get featurize function
        ae = AgentEvaluator.from_layout_name(
            mdp_params=bc_params["mdp_params"],
            env_params=bc_params["env_params"]
        )
        featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
        
        agent = BCAgent(model, bc_params, featurize_fn, agent_index=0)
        
        # Get a state
        state = ae.env.mdp.get_standard_start_state()
        
        # Get action
        action, info = agent.action(state)
        
        # Check action is valid
        self.assertIn(action, Action.ALL_ACTIONS)
        self.assertIn("action_probs", info)
        self.assertEqual(len(info["action_probs"]), Action.NUM_ACTIONS)

    def test_bc_policy_compute_actions(self):
        """Test that BehaviorCloningPolicy can compute actions for batches."""
        from human_aware_rl.imitation.bc_agent import BehaviorCloningPolicy
        
        # Build and save a model
        model = build_bc_model(**self.bc_params)
        save_bc_model(self.model_dir, model, self.bc_params)
        
        # Load model and create policy
        policy = BehaviorCloningPolicy.from_model_dir(self.model_dir)
        
        # Create dummy observations
        obs_batch = np.random.randn(4, *self.bc_params["observation_shape"]).astype(np.float32)
        
        # Compute actions
        actions, states, info = policy.compute_actions(obs_batch)
        
        # Check shapes
        self.assertEqual(len(actions), 4)
        self.assertIn("action_dist_inputs", info)
        self.assertEqual(info["action_dist_inputs"].shape, (4, self.bc_params["action_shape"][0]))


if __name__ == "__main__":
    unittest.main()

