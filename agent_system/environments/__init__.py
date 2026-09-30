"""Active visual environments for VISTA rollouts."""

from vista.environment import ActiveVisualEnvironmentManager

EnvironmentManagerBase = ActiveVisualEnvironmentManager


def make_envs(config):
    if config.env.env_name != "active_visual/crop":
        raise ValueError("VISTA requires active_visual/crop")
    return (
        ActiveVisualEnvironmentManager(config, is_train=True),
        ActiveVisualEnvironmentManager(config, is_train=False),
    )
