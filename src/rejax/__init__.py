from rejax.algos import (
    DQN,
    IQN,
    PPO,
    PQN,
    SAC,
    TD3,
    UCB,
    Algorithm,
)

_algos = {
    "dqn": DQN,
    "iqn": IQN,
    "ppo": PPO,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
    "ucb": UCB,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "get_algo",
    # Algorithms
    "DQN",
    "IQN",
    "PPO",
    "PQN",
    "SAC",
    "TD3",
    "UCB",
]
