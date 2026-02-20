"""
Visualization Script for Watching Trained Agents Play Overcooked.

This script loads trained agents and displays them playing the game
with real-time visualization or saves to GIF.

Usage:
    # RECOMMENDED: Save as GIF and auto-open (no flickering pygame windows)
    python -m human_aware_rl.visualization.play_game --bc_self_play --layout cramped_room --gif
    
    # Quick demo with random agents (no trained models needed)
    python -m human_aware_rl.visualization.play_game --layout cramped_room --random --gif

    # Watch BC agents live (opens pygame window)
    python -m human_aware_rl.visualization.play_game --bc_self_play --layout cramped_room

    # Save GIF to specific path
    python -m human_aware_rl.visualization.play_game --bc_self_play --layout cramped_room --save_gif my_game.gif

    # Available layouts: cramped_room, asymmetric_advantages, coordination_ring, 
    #                   forced_coordination, counter_circuit
"""

import argparse
import os
import time
from typing import Optional, Tuple

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair, RandomAgent, StayAgent
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from human_aware_rl.ppo.configs.paper_configs import PAPER_LAYOUTS, LAYOUT_TO_ENV
from human_aware_rl.imitation.behavior_cloning import BC_SAVE_DIR


def load_bc_agent(
    model_dir: str,
    layout_name: str,
    agent_index: int = 0,
    stochastic: bool = True,
) -> Agent:
    """
    Load a BC agent from a model directory.
    
    Args:
        model_dir: Path to BC model directory
        layout_name: Layout name for featurization
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic action selection
        
    Returns:
        BCAgent instance
    """
    from human_aware_rl.imitation.behavior_cloning import load_bc_model
    from human_aware_rl.imitation.bc_agent import BCAgent
    
    model, bc_params = load_bc_model(model_dir)
    
    # Create agent evaluator for featurization
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    
    featurize_fn = lambda state: ae.env.featurize_state_mdp(state)
    
    return BCAgent(
        model=model,
        bc_params=bc_params,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
    )


def load_ppo_agent(
    checkpoint_dir: str,
    layout_name: str,
    agent_index: int = 0,
    stochastic: bool = True,
) -> Agent:
    """
    Load a JAX PPO agent from a checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        layout_name: Layout name for featurization
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic action selection
        
    Returns:
        JaxPolicyAgent instance
    """
    from human_aware_rl.bridge.jax_agent import JaxPolicyAgent
    
    # Create agent evaluator for featurization
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout_name, "old_dynamics": True},
        env_params={"horizon": 400}
    )
    
    featurize_fn = lambda state: ae.env.lossless_state_encoding_mdp(state)
    
    return JaxPolicyAgent.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        featurize_fn=featurize_fn,
        agent_index=agent_index,
        stochastic=stochastic,
        use_lossless_encoding=True,
    )


def load_agent(
    agent_type: str,
    model_path: Optional[str],
    layout_name: str,
    agent_index: int,
    stochastic: bool = True,
) -> Agent:
    """
    Load an agent based on type.
    
    Args:
        agent_type: Type of agent ('bc', 'ppo', 'random', 'stay')
        model_path: Path to model (not needed for random/stay)
        layout_name: Layout name
        agent_index: Agent index (0 or 1)
        stochastic: Whether to use stochastic actions
        
    Returns:
        Agent instance
    """
    if agent_type == "random":
        return RandomAgent()
    elif agent_type == "stay":
        return StayAgent()
    elif agent_type == "bc":
        if model_path is None:
            raise ValueError("Must provide model_path for BC agent")
        return load_bc_agent(model_path, layout_name, agent_index, stochastic)
    elif agent_type == "ppo":
        if model_path is None:
            raise ValueError("Must provide model_path for PPO agent")
        return load_ppo_agent(model_path, layout_name, agent_index, stochastic)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def play_game(
    layout_name: str,
    agent_0: Agent,
    agent_1: Agent,
    num_games: int = 1,
    horizon: int = 400,
    fps: int = 10,
    render: bool = True,
    verbose: bool = True,
    save_gif: Optional[str] = None,
    save_frames_dir: Optional[str] = None,
) -> dict:
    """
    Play games between two agents with visualization.
    
    Args:
        layout_name: Layout name
        agent_0: First agent
        agent_1: Second agent
        num_games: Number of games to play
        horizon: Episode length
        fps: Frames per second for visualization
        render: Whether to render the game
        verbose: Whether to print scores
        save_gif: Path to save animated GIF (optional)
        save_frames_dir: Directory to save individual frames (optional)
        
    Returns:
        Dictionary with game results
    """
    # Create environment
    mdp = OvercookedGridworld.from_layout_name(
        layout_name,
        old_dynamics=True
    )
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    
    # Track frames for GIF
    frames = []
    
    # Create visualizer if rendering or saving
    should_visualize = render or save_gif or save_frames_dir
    if should_visualize:
        try:
            import pygame
            
            # Use dummy video driver if we're not displaying to screen
            # This prevents pygame from creating visible windows
            if not render:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
            pygame.init()
            visualizer = StateVisualizer()
        except ImportError:
            print("Warning: pygame not available, disabling visualization")
            render = False
            save_gif = None
            save_frames_dir = None
            visualizer = None
    else:
        visualizer = None
    
    all_returns = []
    all_lengths = []
    
    for game_idx in range(num_games):
        # Reset environment and agents
        env.reset()
        agent_0.reset()
        agent_1.reset()
        
        total_reward = 0
        step = 0
        done = False
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game {game_idx + 1}/{num_games}")
            print(f"{'='*50}")
        
        while not done:
            state = env.state
            
            # Render current state
            if visualizer is not None:
                # Save frame for GIF if requested
                if save_gif or save_frames_dir:
                    try:
                        import pygame
                        from PIL import Image
                        
                        # Get the rendered pygame surface
                        surface = visualizer.render_state(
                            state=state,
                            grid=mdp.terrain_mtx,
                        )
                        
                        # Convert pygame surface to PIL Image
                        surface_str = pygame.image.tostring(surface, 'RGB')
                        pil_img = Image.frombytes(
                            'RGB', 
                            surface.get_size(), 
                            surface_str
                        )
                        frames.append(pil_img)
                        
                        if save_frames_dir:
                            os.makedirs(save_frames_dir, exist_ok=True)
                            frame_path = os.path.join(
                                save_frames_dir, 
                                f"game{game_idx:02d}_step{step:04d}.png"
                            )
                            pil_img.save(frame_path)
                    except ImportError as e:
                        print(f"Warning: Cannot save frames - {e}")
                        save_gif = None
                        save_frames_dir = None
                
                # Display in window if requested
                if render:
                    visualizer.display_rendered_state(
                        state=state,
                        grid=mdp.terrain_mtx,
                        ipython_display=False,
                        window_display=True
                    )
                    time.sleep(1.0 / fps)
            
            # Get actions from agents
            action_0, _ = agent_0.action(state)
            action_1, _ = agent_1.action(state)
            
            joint_action = (action_0, action_1)
            
            # Step environment
            next_state, reward, done, info = env.step(joint_action)
            
            total_reward += reward
            step += 1
            
            if verbose and reward > 0:
                print(f"Step {step}: Reward +{reward} (Total: {total_reward})")
        
        all_returns.append(total_reward)
        all_lengths.append(step)
        
        if verbose:
            print(f"\nGame {game_idx + 1} finished!")
            print(f"Total reward: {total_reward}")
            print(f"Episode length: {step}")
    
    # Save GIF if requested
    if save_gif and frames:
        try:
            from PIL import Image
            print(f"\nSaving GIF to {save_gif}...")
            # Save as animated GIF
            frames[0].save(
                save_gif,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),  # Duration per frame in ms
                loop=0  # Infinite loop
            )
            print(f"GIF saved with {len(frames)} frames")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    # Cleanup
    if render or save_gif or save_frames_dir:
        try:
            import pygame
            pygame.quit()
        except:
            pass
    
    results = {
        "mean_reward": np.mean(all_returns),
        "std_reward": np.std(all_returns),
        "min_reward": np.min(all_returns),
        "max_reward": np.max(all_returns),
        "mean_length": np.mean(all_lengths),
        "all_returns": all_returns,
        "all_lengths": all_lengths,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"Games played: {num_games}")
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Min/Max reward: {results['min_reward']}/{results['max_reward']}")
    
    return results


def find_latest_checkpoint(base_dir: str, pattern: str) -> Optional[str]:
    """Find the latest checkpoint matching a pattern."""
    if not os.path.exists(base_dir):
        return None
    
    for exp_name in os.listdir(base_dir):
        if pattern in exp_name:
            exp_dir = os.path.join(base_dir, exp_name)
            if os.path.isdir(exp_dir):
                checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint")]
                if checkpoints:
                    latest = sorted(checkpoints)[-1]
                    return os.path.join(exp_dir, latest)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Watch trained agents play Overcooked",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Layout
    parser.add_argument(
        "--layout",
        type=str,
        default="cramped_room",
        help="Layout name"
    )
    
    # Agent 0 configuration
    parser.add_argument(
        "--agent0_type",
        type=str,
        default="bc",
        choices=["bc", "ppo", "random", "stay"],
        help="Type of agent 0"
    )
    parser.add_argument(
        "--agent0_path",
        type=str,
        default=None,
        help="Path to agent 0 model"
    )
    
    # Agent 1 configuration
    parser.add_argument(
        "--agent1_type",
        type=str,
        default=None,
        help="Type of agent 1 (default: same as agent 0)"
    )
    parser.add_argument(
        "--agent1_path",
        type=str,
        default=None,
        help="Path to agent 1 model (default: same as agent 0)"
    )
    
    # Shortcuts
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random agents (quick demo)"
    )
    parser.add_argument(
        "--bc_self_play",
        action="store_true",
        help="Use BC self-play with default paths"
    )
    parser.add_argument(
        "--ppo_self_play",
        action="store_true",
        help="Use PPO self-play with default paths"
    )
    
    # Game settings
    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="Number of games to play"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Episode length"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for visualization"
    )
    
    # Rendering options
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable visual rendering"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic (argmax) actions instead of sampling"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--save_gif",
        type=str,
        default=None,
        help="Save game as animated GIF to this path"
    )
    parser.add_argument(
        "--save_frames",
        type=str,
        default=None,
        help="Save individual frames to this directory"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Save as GIF and open when done (no live pygame window)"
    )
    
    args = parser.parse_args()
    
    # Handle --gif shortcut: save to temp file and open when done
    if args.gif:
        args.no_render = True
        if args.save_gif is None:
            import tempfile
            args.save_gif = os.path.join(tempfile.gettempdir(), f"overcooked_{args.layout}.gif")
    
    # Handle layout mapping
    layout = args.layout
    if layout in LAYOUT_TO_ENV:
        env_layout = LAYOUT_TO_ENV[layout]
    else:
        env_layout = layout
    
    stochastic = not args.deterministic
    
    # Handle shortcuts
    if args.random:
        args.agent0_type = "random"
        args.agent1_type = "random"
    elif args.bc_self_play:
        args.agent0_type = "bc"
        args.agent1_type = "bc"
        if args.agent0_path is None:
            args.agent0_path = os.path.join(BC_SAVE_DIR, "train", layout)
    elif args.ppo_self_play:
        args.agent0_type = "ppo"
        args.agent1_type = "ppo"
        if args.agent0_path is None:
            # Try to find a PPO checkpoint
            args.agent0_path = find_latest_checkpoint("results/ppo_sp", layout)
            if args.agent0_path is None:
                args.agent0_path = find_latest_checkpoint("results/ppo_bc", layout)
    
    # Default agent 1 to same as agent 0
    if args.agent1_type is None:
        args.agent1_type = args.agent0_type
    if args.agent1_path is None:
        args.agent1_path = args.agent0_path
    
    # Print configuration
    if not args.quiet:
        print("="*60)
        print("Overcooked Agent Visualization")
        print("="*60)
        print(f"Layout: {layout} -> {env_layout}")
        print(f"Agent 0: {args.agent0_type}" + (f" ({args.agent0_path})" if args.agent0_path else ""))
        print(f"Agent 1: {args.agent1_type}" + (f" ({args.agent1_path})" if args.agent1_path else ""))
        print(f"Games: {args.num_games}")
        print(f"Horizon: {args.horizon}")
        print(f"FPS: {args.fps}")
        print(f"Stochastic: {stochastic}")
        print("="*60)
    
    # Load agents
    try:
        agent_0 = load_agent(
            args.agent0_type,
            args.agent0_path,
            env_layout,
            agent_index=0,
            stochastic=stochastic,
        )
    except Exception as e:
        print(f"Error loading agent 0: {e}")
        if args.agent0_type in ["bc", "ppo"]:
            print(f"Make sure you have trained models at: {args.agent0_path}")
            print(f"\nTo train BC models: python -m human_aware_rl.imitation.train_bc_models --all_layouts")
            print(f"To train PPO models: python -m human_aware_rl.ppo.train_ppo_sp --all_layouts")
        return
    
    try:
        agent_1 = load_agent(
            args.agent1_type,
            args.agent1_path,
            env_layout,
            agent_index=1,
            stochastic=stochastic,
        )
    except Exception as e:
        print(f"Error loading agent 1: {e}")
        return
    
    # Play games
    results = play_game(
        layout_name=env_layout,
        agent_0=agent_0,
        agent_1=agent_1,
        num_games=args.num_games,
        horizon=args.horizon,
        fps=args.fps,
        render=not args.no_render,
        verbose=not args.quiet,
        save_gif=args.save_gif,
        save_frames_dir=args.save_frames,
    )
    
    # Open GIF if --gif mode was used
    if args.gif and args.save_gif and os.path.exists(args.save_gif):
        print(f"\nOpening GIF: {args.save_gif}")
        import subprocess
        import sys
        
        # Open with system default viewer
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", args.save_gif])
        elif sys.platform == "linux":
            subprocess.run(["xdg-open", args.save_gif])
        elif sys.platform == "win32":
            os.startfile(args.save_gif)
    
    return results


if __name__ == "__main__":
    main()

