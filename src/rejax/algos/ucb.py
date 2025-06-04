import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp
from typing import NamedTuple

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    BanditMixin,
    StoreTrajectoriesMixin,
)
from rejax.networks import TabularUCB
from rejax.buffers import Minibatch


class UCB(BanditMixin, StoreTrajectoriesMixin, Algorithm):
    agent: nn.Module = struct.field(pytree_node=False, default=None)

    def make_act(self, ts):
        def act(obs, rng):
            obs = jnp.expand_dims(obs, 0)

            action = self.agent.apply(ts.agent_ts.params, obs, method="act")

            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        action_space = env.action_space(env_params)
        agent_kwargs = config.pop("agent_kwargs", {})

        agent = TabularUCB(
            num_arms=action_space.n,
            confidence=agent_kwargs.get("confidence", 1.0),
        )

        return {"agent": agent,}

    @register_init
    def initialize_network_params(self, rng):
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])
        rng, rng_agent = jax.random.split(rng, 2)

        agent_params = self.agent.init(rng_agent, obs_ph)

        agent_ts = TrainState.create(apply_fn=(), params=agent_params, tx=optax.identity())
        return {"agent_ts": agent_ts,}

    def train_iteration(self, ts):
        ts, transition = self.collect_transition(ts)
        ts = ts.replace(store_buffer=ts.store_buffer.extend(transition))

        # Sample minibatch
        rng, rng_sample = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        # Update agent
        ts = self.update(ts, transition)
        return ts

    def collect_transition(self, ts):
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_policy(rng):
            last_obs = ts.last_obs

            return self.agent.apply(
                ts.agent_ts.params, last_obs, method="act"
            )

        actions = sample_policy(rng_action)

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, 1)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )

        transition = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + 1,
        )
        return ts, transition

    def update(self, ts, transition):
        action_dim = ts.agent_ts.params["params"]["counts"].shape[1]
        update_mask = jnp.eye(action_dim)[transition.action]

        # Update UCB parameters
        counts = ts.agent_ts.params["params"]["counts"] + update_mask
        q_values = ts.agent_ts.params["params"]["q_values"] + update_mask * (
            transition.reward - ts.agent_ts.params["params"]["q_values"]
        ) / jnp.clip(counts, min=1)
        
        timesteps = ts.agent_ts.params["params"]["timesteps"] + 1

        updates = {
            "params": {
                "q_values": q_values,
                "counts": counts,
                "timesteps": timesteps,
            }
        }

        # Apply update
        agent_ts = ts.agent_ts.replace(
            params=updates,
        )
        ts = ts.replace(agent_ts=agent_ts)
        return ts
