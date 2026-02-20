"""
WebPPL Bridge for Overcooked AI

This module provides a bridge between Python and WebPPL for cognitive
modeling of agents in Overcooked. WebPPL offers:

1. Memoization patterns (useful for caching expensive computations)
2. Inference algorithms like enumerate, rejection, MCMC
3. Syntax closer to traditional cognitive science models
4. Easy specification of RSA-style models

Usage:
1. Define your model in WebPPL (see examples below)
2. Use this bridge to run inference and get action distributions
3. Export learned parameters back to Python

Requirements:
- Node.js installed
- webppl package: npm install -g webppl

Example WebPPL Model (Rational Speech Acts style):
```javascript
// Agent chooses actions to maximize utility
var agent = function(state, softmaxBeta) {
  return Infer({method: 'enumerate'}, function() {
    var action = uniformDraw(actions);
    var utility = getUtility(state, action);
    factor(softmaxBeta * utility);
    return action;
  });
};
```

References:
- WebPPL: http://webppl.org
- Probabilistic Models of Cognition: https://probmods.org
"""

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


@dataclass  
class WebPPLConfig:
    """Configuration for WebPPL models."""
    
    # Path to WebPPL executable (usually 'webppl' if installed globally)
    webppl_path: str = "webppl"
    
    # Inference method
    inference_method: str = "enumerate"  # enumerate, rejection, MCMC
    
    # Number of samples (for rejection/MCMC)
    num_samples: int = 1000
    
    # Model parameters
    softmax_beta: float = 1.0  # Rationality parameter
    
    # Timeout for WebPPL execution (seconds)
    timeout: int = 30


class WebPPLModel:
    """
    Base class for WebPPL models.
    
    Subclass this and implement:
    - get_webppl_code(): Returns the WebPPL code as a string
    - encode_state(): Converts Overcooked state to WebPPL-compatible format
    """
    
    def __init__(self, config: WebPPLConfig = None):
        self.config = config or WebPPLConfig()
        self._check_webppl()
    
    def _check_webppl(self):
        """Check if WebPPL is installed."""
        try:
            result = subprocess.run(
                [self.config.webppl_path, "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                print("WARNING: WebPPL not found. Install with: npm install -g webppl")
        except Exception as e:
            print(f"WARNING: Could not check WebPPL: {e}")
    
    def get_webppl_code(self) -> str:
        """Override this to return your WebPPL model code."""
        raise NotImplementedError
    
    def encode_state(self, state) -> Dict[str, Any]:
        """Override this to encode Overcooked state for WebPPL."""
        raise NotImplementedError
    
    def run_inference(self, state) -> Dict[str, float]:
        """
        Run WebPPL inference and return action distribution.
        
        Returns:
            Dict mapping action names to probabilities
        """
        # Encode state
        state_json = json.dumps(self.encode_state(state))
        
        # Build full WebPPL program
        program = f"""
        var state = {state_json};
        var config = {json.dumps(self.config.__dict__)};
        
        {self.get_webppl_code()}
        
        // Run inference
        var actionDist = agent(state, config.softmax_beta);
        actionDist;
        """
        
        # Write to temp file and run
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wppl', delete=False) as f:
            f.write(program)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [self.config.webppl_path, temp_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"WebPPL error: {result.stderr}")
            
            # Parse output (WebPPL outputs JSON)
            output = json.loads(result.stdout)
            
            # Convert to action distribution
            if isinstance(output, dict) and "probs" in output:
                return dict(zip(output["support"], output["probs"]))
            else:
                return output
                
        finally:
            os.unlink(temp_path)


class SoftmaxRationalModel(WebPPLModel):
    """
    Simple softmax-rational agent model in WebPPL.
    
    This is a direct implementation of:
        P(action | state) ∝ exp(β * U(state, action))
    """
    
    def __init__(self, utility_fn: str = None, config: WebPPLConfig = None):
        """
        Args:
            utility_fn: WebPPL code for utility function.
                        Should define: var getUtility = function(state, action) {...}
            config: WebPPL configuration
        """
        super().__init__(config)
        
        self.utility_fn = utility_fn or """
        // Default utility: random (no preferences)
        var getUtility = function(state, action) {
            return 0;
        };
        """
    
    def get_webppl_code(self) -> str:
        return f"""
        // Available actions in Overcooked
        var actions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'STAY', 'INTERACT'];
        
        // Utility function
        {self.utility_fn}
        
        // Softmax-rational agent
        var agent = function(state, beta) {{
            return Infer({{method: 'enumerate'}}, function() {{
                var action = uniformDraw(actions);
                var utility = getUtility(state, action);
                factor(beta * utility);
                return action;
            }});
        }};
        """
    
    def encode_state(self, state) -> Dict[str, Any]:
        """
        Encode Overcooked state for WebPPL.
        
        This is a simplified encoding - extend as needed.
        """
        # Extract key features from state
        players = state.players
        
        encoded = {
            "player0": {
                "position": list(players[0].position),
                "orientation": list(players[0].orientation),
                "held_object": str(players[0].held_object) if players[0].held_object else None,
            },
            "player1": {
                "position": list(players[1].position),
                "orientation": list(players[1].orientation),
                "held_object": str(players[1].held_object) if players[1].held_object else None,
            },
        }
        
        return encoded


class GoalInferenceModel(WebPPLModel):
    """
    Goal inference model using WebPPL's memoization.
    
    This model uses the "Planning as Inference" approach:
    1. Infer what goal the agent is pursuing
    2. Compute optimal actions for that goal
    3. Return action distribution
    
    Uses memo pattern for caching expensive planning computations.
    """
    
    def __init__(self, goals: List[str] = None, config: WebPPLConfig = None):
        """
        Args:
            goals: List of possible goals (e.g., ['get_onion', 'serve_soup'])
            config: WebPPL configuration
        """
        super().__init__(config)
        
        self.goals = goals or [
            "get_onion",
            "get_tomato", 
            "deliver_to_pot",
            "serve_soup",
            "wait",
        ]
    
    def get_webppl_code(self) -> str:
        goals_json = json.dumps(self.goals)
        
        return f"""
        var actions = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'STAY', 'INTERACT'];
        var goals = {goals_json};
        
        // Memoized goal-conditioned utility
        var goalUtility = mem(function(goal, state, action) {{
            // Simplified utility - extend based on goal semantics
            if (goal === 'get_onion') {{
                // Reward moving toward onion station
                return action === 'INTERACT' ? 1.0 : 0.0;
            }} else if (goal === 'serve_soup') {{
                return action === 'INTERACT' ? 1.0 : 0.0;
            }} else {{
                return 0.0;
            }}
        }});
        
        // Goal inference from state
        var inferGoal = mem(function(state) {{
            return Infer({{method: 'enumerate'}}, function() {{
                var goal = uniformDraw(goals);
                
                // Prior over goals based on state
                var goalPrior = goal === 'wait' ? 0.1 : 0.9 / (goals.length - 1);
                factor(Math.log(goalPrior));
                
                return goal;
            }});
        }});
        
        // Hierarchical agent
        var agent = function(state, beta) {{
            return Infer({{method: 'enumerate'}}, function() {{
                // Sample goal from inferred distribution
                var goalDist = inferGoal(state);
                var goal = sample(goalDist);
                
                // Sample action conditioned on goal
                var action = uniformDraw(actions);
                var utility = goalUtility(goal, state, action);
                factor(beta * utility);
                
                return action;
            }});
        }};
        """
    
    def encode_state(self, state) -> Dict[str, Any]:
        """Encode state for goal inference."""
        players = state.players
        
        return {
            "player0": {
                "position": list(players[0].position),
                "held_object": str(players[0].held_object) if players[0].held_object else None,
            },
            "player1": {
                "position": list(players[1].position),
                "held_object": str(players[1].held_object) if players[1].held_object else None,
            },
            "time_step": getattr(state, "timestep", 0),
        }


class WebPPLAgent(Agent):
    """
    Agent that uses a WebPPL model for action selection.
    
    Note: This is slower than neural network models due to
    subprocess overhead. Best used for:
    1. Prototyping cognitive models
    2. Interpretability analysis  
    3. Generating "rational" baselines
    """
    
    def __init__(
        self,
        model: WebPPLModel,
        agent_index: int = 0,
        stochastic: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.agent_index = agent_index
        self.stochastic = stochastic
        
        # Action name mapping
        self.action_name_to_idx = {
            'NORTH': 0, 'SOUTH': 1, 'EAST': 2, 'WEST': 3,
            'STAY': 4, 'INTERACT': 5,
        }
    
    def action(self, state) -> Tuple[Any, Dict]:
        """Select action using WebPPL inference."""
        try:
            # Run WebPPL inference
            action_dist = self.model.run_inference(state)
            
            # Convert to numpy array
            probs = np.zeros(len(Action.ALL_ACTIONS))
            for action_name, prob in action_dist.items():
                if action_name in self.action_name_to_idx:
                    probs[self.action_name_to_idx[action_name]] = prob
            
            # Normalize
            probs = probs / (probs.sum() + 1e-8)
            
            # Select action
            if self.stochastic:
                action_idx = np.random.choice(len(probs), p=probs)
            else:
                action_idx = np.argmax(probs)
            
            action = Action.INDEX_TO_ACTION[action_idx]
            
            return action, {"action_probs": probs}
            
        except Exception as e:
            # Fallback to random action
            print(f"WebPPL error: {e}, using random action")
            action_idx = np.random.randint(len(Action.ALL_ACTIONS))
            return Action.INDEX_TO_ACTION[action_idx], {"error": str(e)}
    
    def reset(self):
        pass


# Example WebPPL utility functions for Overcooked

OVERCOOKED_UTILITIES = """
// Utility function for onion soup task
var getUtility = function(state, action) {
    var player = state.player0;  // Or state.player1
    
    // Check if holding something
    var holding = player.held_object;
    
    // Simple heuristic utilities
    if (holding === null) {
        // Want to pick up ingredient
        if (action === 'INTERACT') return 1.0;
        return 0.1;  // Moving is okay
    } else if (holding.includes('onion')) {
        // Want to deliver to pot
        if (action === 'INTERACT') return 1.0;
        return 0.1;
    } else if (holding.includes('soup')) {
        // Want to serve
        if (action === 'INTERACT') return 2.0;  // High reward for serving
        return 0.1;
    }
    
    return 0.0;
};
"""


def create_webppl_agent(
    model_type: str = "softmax_rational",
    softmax_beta: float = 1.0,
    **kwargs,
) -> WebPPLAgent:
    """
    Factory function to create WebPPL agents.
    
    Args:
        model_type: "softmax_rational" or "goal_inference"
        softmax_beta: Rationality parameter
        **kwargs: Additional arguments for the model
        
    Returns:
        WebPPLAgent instance
    """
    config = WebPPLConfig(softmax_beta=softmax_beta)
    
    if model_type == "softmax_rational":
        utility_fn = kwargs.get("utility_fn", OVERCOOKED_UTILITIES)
        model = SoftmaxRationalModel(utility_fn=utility_fn, config=config)
    elif model_type == "goal_inference":
        goals = kwargs.get("goals", None)
        model = GoalInferenceModel(goals=goals, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return WebPPLAgent(model, **kwargs)


if __name__ == "__main__":
    # Test WebPPL bridge
    print("Testing WebPPL bridge...")
    
    config = WebPPLConfig(softmax_beta=2.0)
    model = SoftmaxRationalModel(utility_fn=OVERCOOKED_UTILITIES, config=config)
    
    # Test with dummy state
    dummy_state = {
        "player0": {"position": [1, 1], "held_object": None},
        "player1": {"position": [3, 3], "held_object": None},
    }
    
    try:
        # Note: This will only work if WebPPL is installed
        action_dist = model.run_inference(type('State', (), dummy_state)())
        print(f"Action distribution: {action_dist}")
    except Exception as e:
        print(f"WebPPL not available: {e}")
        print("Install with: npm install -g webppl")
