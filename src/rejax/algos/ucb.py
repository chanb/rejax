import chex
import gymnax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    OnPolicyMixin,
)
from rejax.buffers import Minibatch


class UCB(OnPolicyMixin, Algorithm):
    q_table: nn.Module = struct.field(pytree_node=False, default=None)
    counts: nn.Module = struct.field(pytree_node=False, default=None)
    timesteps: int = struct.field(pytree_node=False, default=0)
    num_epochs: int = struct.field(pytree_node=False, default=8)

    def make_act(self, ts):
        def act(obs, rng):
            obs = jnp.expand_dims(obs, 0)
            
            # TODO: FIX THIS
            action = None
            
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config, env, env_params):
        # TODO: Fix this
        action_space = env.action_space(env_params)

        q_table = None
        counts = None
        timesteps = None

        return {
            "q_table": q_table,
            "counts": counts,
            "timesteps": timesteps,
        }

    @register_init
    def initialize_network_params(self, rng):
        # TODO: Fix this
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        obs_ph = jnp.empty([1, *self.env.observation_space(self.env_params).shape])

        actor_params = self.actor.init(rng_actor, obs_ph, rng_actor)
        critic_params = self.critic.init(rng_critic, obs_ph)

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        actor_ts = TrainState.create(apply_fn=(), params=actor_params, tx=tx)
        critic_ts = TrainState.create(apply_fn=(), params=critic_params, tx=tx)
        return {"actor_ts": actor_ts, "critic_ts": critic_ts}

    def train_iteration(self, ts):
        # TODO: Fix this
        ts, batch = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        # Sample minibatch
        rng, rng_sample = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
        if self.normalize_observations:
            minibatch = minibatch._replace(
                obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
            )

        # Update network
        ts = self.update(ts, minibatch)
        return ts

    def collect_transitions(self, ts):
        # Sample actions
        # TODO: Fix this
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_policy(rng):
            last_obs = ts.last_obs

            return self.agent.apply(
                ts.q_ts.params, last_obs, rng, method="act"
            )

        actions = sample_policy(rng_action)

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(
            rng_steps, ts.env_state, actions, self.env_params
        )

        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update(self, ts, batch):
        # TODO: Fix this
        return ts
