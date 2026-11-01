"""SeaSwarm: Bayesian search for a person in water with a drone and USV swarm."""

import os

# NOTE: LiteLLM otherwise downloads a model price list from GitHub every time it is imported.
# The project runs locally and offline, so use the copy bundled with LiteLLM (overridable).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

__version__ = "0.0.1"
