"""
Shared defaults for canonical PPO run-registry loading.
"""

DEFAULT_RUN_NAME_TEMPLATES = {
    "ppo_sp": "ppo_sp__layout-{layout}",
    "ppo_bc": "ppo_bc__partner-bc_train__layout-{layout}",
    "ppo_hp": "ppo_hp__layout-{layout}",
}

DEFAULT_AGENT_DIRS = {
    "ppo_sp": "ppo_agent",
    "ppo_bc": "ppo_bc_agent",
    "ppo_hp": "ppo_hp_agent",
}
