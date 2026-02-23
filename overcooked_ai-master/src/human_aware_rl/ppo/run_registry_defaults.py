"""
Shared defaults for canonical PPO run-registry loading.
"""

from human_aware_rl.ppo.run_paths import format_run_template


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


def get_default_run_template(source: str, partner_type: str = "bc_train") -> str:
    """
    Return the canonical run-name template for a source.

    For PPO_BC, supports partner-specific variants while keeping the canonical
    default template rooted in DEFAULT_RUN_NAME_TEMPLATES.
    """
    if source not in DEFAULT_RUN_NAME_TEMPLATES:
        raise KeyError(f"Unknown source '{source}'")
    template = DEFAULT_RUN_NAME_TEMPLATES[source]
    if source == "ppo_bc" and partner_type != "bc_train":
        template = template.replace("partner-bc_train", f"partner-{partner_type}")
    return template


def get_default_run_name(source: str, layout: str, partner_type: str = "bc_train") -> str:
    """Return canonical run name with {layout} resolved."""
    template = get_default_run_template(source=source, partner_type=partner_type)
    return format_run_template(template, layout)


def get_default_agent_dir(source: str) -> str:
    """Return canonical agent directory name for a source."""
    if source not in DEFAULT_AGENT_DIRS:
        raise KeyError(f"Unknown source '{source}'")
    return DEFAULT_AGENT_DIRS[source]
