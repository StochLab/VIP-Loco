from copy import deepcopy
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map as jax_tree_map
import distrax

import torch
import torch.nn as nn

from .torch2jax import t2j, j2t

from typing import Dict

import functools


class JaxMPC():
    def __init__(self, world_model: Dict[str, nn.Module], expert_actor: nn.Module,
                wm_horizon: int = 5,
                wm_freq: int = 5):
        self.wm_freq = wm_freq
        self.wm_horizon = wm_horizon

        self.obs_cmd_index = 3 + 3 + 3 # base lin vel, ang vel, projected grav

        # Modules & Functions
        self.encoder = t2j(world_model['encoder'])(state_dict={k: t2j(v) for k, v in world_model['encoder'].named_parameters()}).forward # type: ignore
        self.decoder = t2j(world_model['decoder'])(state_dict={k: t2j(v) for k, v in world_model['decoder'].named_parameters()}).forward # type: ignore

        # KinoDynamicCoM RSSM
        self.dynamics = t2j(world_model['dynamics'])(state_dict={k: t2j(v) for k, v in world_model['dynamics'].named_parameters()}) # type: ignore

        # Distributions. Will use only the mean
        self.pi = t2j(world_model['actor'])(state_dict={k: t2j(v) for k, v in world_model['actor'].named_parameters()}).forward # type: ignore
        self.value = t2j(world_model['value'])(state_dict={k: t2j(v) for k, v in world_model['value'].named_parameters()}).forward # type: ignore
        self.reward = t2j(world_model['reward'])(state_dict={k: t2j(v) for k, v in world_model['reward'].named_parameters()}).forward # type: ignore

        # Expert actor. Will use only the mean
        self.expert_actor = t2j(expert_actor)(state_dict={k: t2j(v) for k, v in expert_actor.named_parameters()}).forward # type: ignore

        self.act_dim: int = self.dynamics._actions_dim # type: ignore
        self.planning_horizon = 8
        self.gamma = 0.99 # Taken from config
        self.num_pi_traj = 20 # Number of initial trajectories
        self.M = 500 # Number of samples per iteration
        self.N = 6 # Number of iterations for MPC
        self.K = 100 # Number of elite samples
        self.temp = 0.1 # Temperature for weighting the rewards
        self.smooth = 0.9 # Smoothing factor for the mean and std_dev

    def init_wm_state(self, wm_obs):
        """
            Initialize the world model state
        """
        wm_obs_jax = jax_tree_map(t2j, wm_obs)

        is_first = wm_obs_jax['is_first']
        embed = self.encoder(wm_obs_jax)
        wm_state =  None

        # Initialize the world model state
        wm_state = self.dynamics.check_state_reset_with_masks(is_first, embed, wm_state)

        return jax_tree_map(j2t, wm_state)

    def plan(self, rng, mpc_state: Dict):
        """
            Plan for the next action
            Args:
                rng: JAX random number generator
                prev_mpc_state: Previous MPC state, used for warm starting the MPC
                data: Input data containing observations and other necessary information. 
                    Contains 1. Observations from the environment 2. World model observations 3. History of observations
        """
        # convert torch input to jax using pytrees
        mpc_state_jax = jax_tree_map(t2j, mpc_state)

        expert_action_jax, mpc_state_jax = self.forward(
                rng, mpc_state_jax
            )

        expert_action_torch = j2t(expert_action_jax)
        mpc_state = jax_tree_map(j2t, mpc_state_jax)

        return expert_action_torch, mpc_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def forward(self, rng, mpc_state: Dict):
        # Get a warm start using expert actor

        wm_feature, wm_state_post = jax.lax.cond(
            mpc_state['step'] % self.wm_freq == 0,
            lambda x: self.get_wm_feature(x['wm_obs'], x['wm_state']), 
            lambda x: (x['wm_feature'], x['wm_state']),
            mpc_state
        )        

        trajectory_history = mpc_state['trajectory_history']
        history = jnp.reshape(trajectory_history, (trajectory_history.shape[0], -1)) # (B, history_dim)

        # Get the expert action
        expert_action = self.expert_actor(mpc_state['cmd'], history, wm_feature) # (B, nu)

        PLAN = True
        if PLAN:
            rng, rng_warm = jax.random.split(rng)
            warm_start = self.warm_start_expert(rng_warm, mpc_state['cmd'], expert_action, wm_state_post, trajectory_history)
            rng, rng_solve = jax.random.split(rng)
            planned_action, prev_mean, data_com_pos = self.solve_mpc(
                rng_solve, warm_start, mpc_state['prev_mean'], wm_state_post)
            mpc_state['prev_mean'] = prev_mean # Update the previous mean for next iteration
            mpc_state['data_com_pos'] = data_com_pos

        else:
            # If not planning, just return the first expert action
            planned_action = expert_action        


        mpc_state['wm_feature'] = wm_feature
        mpc_state['wm_state'] = wm_state_post
        mpc_state['step'] += 1

        return planned_action, mpc_state

    def get_wm_feature(self, wm_obs, wm_state):
        wm_embed = self.encoder(wm_obs) # (B, ...)
        wm_state = self.dynamics.check_state_reset_with_masks(
                wm_obs['is_first'], wm_embed, wm_state
            ) # (B, ...)

        y = []
        for t in range(self.wm_horizon):
            action = self.pi(self.dynamics.get_feat(wm_state))['action_mean']

            if t == 0:
                action = self.dynamics.check_action_reset_with_masks(
                        wm_obs['is_first'], action
                    )
                wm_state = self.dynamics.img_step(wm_state, action)
                wm_state_post = self.dynamics.post_step(wm_state['deter'], wm_embed)
                wm_deter_feat = self.dynamics.get_deter_feat(wm_state_post)
            else:
                wm_state = self.dynamics.img_step(wm_state, action)
            
            y.append(self.dynamics.get_com_joint_feat(wm_state)) # (B, ...)

        y = jnp.concatenate(y, axis=-1) # (B, ...)

        return jnp.concatenate([wm_deter_feat, y], axis=-1), wm_state_post

    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0))
    def warm_start_expert(self, rng, cmd: jax.Array, expert_action: jax.Array, 
                            wm_state: Dict, trajectory_history: jax.Array):
        # expert_action: (nu)
        # We want: (num_pi_traj, horizon, nu)

        # expand all to be (num_pi_traj, ...)
        expand_fn = lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self.num_pi_traj, axis=0)
        wm_state = jax_tree_map(expand_fn, deepcopy(wm_state)) # (num_pi_traj, ...)
        trajectory_history = jax_tree_map(expand_fn, deepcopy(trajectory_history)) # (num_pi_traj, history_len, prop_dim)
        expert_action = jax_tree_map(expand_fn, deepcopy(expert_action)) # (num_pi_traj, nu)
        cmd = jax_tree_map(expand_fn, deepcopy(cmd)) # (num_pi_traj, cmd_dim)

        y = [] # collected all the actions
        y.append(jnp.expand_dims(expert_action, axis=1)) # (num_pi_traj, 1, nu)

        for t in range(1, self.planning_horizon):
            prev_action = deepcopy(expert_action) # (num_pi_traj, nu)

            # decode and get the observations
            decode_obs = self.decoder(self.dynamics.get_feat(wm_state)) # (num_pi_traj, ...)

            obs_without_command = jnp.concatenate([
                decode_obs['prop_mean'][..., :self.obs_cmd_index],
                decode_obs['prop_mean'][..., self.obs_cmd_index + cmd.shape[-1]:],
                prev_action], axis=-1) # (num_pi_traj, prop_dim - cmd_dim + nu)

            trajectory_history = jnp.concatenate([
                    trajectory_history[:, 1:], # (num_pi_traj, history_len - 1, prop_dim - cmd_dim + nu)
                    jnp.expand_dims(obs_without_command, axis=1) # (num_pi_traj, 1, prop_dim - cmd_dim + nu)
                ], axis=1)

            history = jnp.reshape(trajectory_history, (trajectory_history.shape[0], -1)) # (num_pi_traj, history_dim)

            # get wm_feature
            wm_feature, wm_state = self.get_wm_feature(
                {'prop': decode_obs['prop_mean'],
                 'image': decode_obs['image_mean'],
                 'is_first': jnp.zeros((self.num_pi_traj,))}, # Assuming no first step for warm start
                wm_state)

            # get the action from the expert actor
            expert_action = self.expert_actor(cmd, history, wm_feature) # (num_pi_traj, nu)

            # Add noise to the action
            rng, rng_noise = jax.random.split(rng)
            expert_action += jax.random.normal(rng_noise, expert_action.shape) * 0.1 # Add some noise

            y.append(jnp.expand_dims(expert_action, axis=1)) # (num_pi_traj, t + 1, nu)

        warm_start = jnp.concatenate(y, axis=1) # (num_pi_traj, horizon, nu)

        return warm_start

    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
    def solve_mpc(self, rng, warm_start: jax.Array, prev_mean: jax.Array, first_wm_state: Dict[str, jax.Array]):
        # warm_start: (num_pi_traj, horizon, nu)
        # wm_state: (num_pi_traj, ...)

        # Initialize the mean and std_dev
        actor_mean = jnp.mean(warm_start, axis=0) # (horizon, nu)
        mean = actor_mean.copy()
        std_dev = warm_start.std(axis=0) # (horizon, nu)

        def rew_k_fn(k: int, z_k: Dict[str, jax.Array], act_k: jax.Array):
            r_k = self.reward(
                    jnp.concatenate([self.dynamics.get_feat(z_k), act_k], axis=-1)
                )['reward_mean']
            v_k = self.value(self.dynamics.get_feat(z_k))['value_mean']

            reward_k = jnp.where((k == self.planning_horizon - 1), v_k, r_k)
            
            return (self.gamma**k) * reward_k

        def total_reward(u_sample):
            total_reward = 0.
            z_k = first_wm_state

            # get com_positions
            data_com_pos = []
            data_com_pos.append(jnp.expand_dims(z_k['com_joint'][..., :3], axis=0)) # (1, 3)

            for k in range(0, self.planning_horizon):
                z_k = self.dynamics.img_step(z_k, u_sample[k])
                total_reward += rew_k_fn(k, z_k, u_sample[k])

                # get com_positions
                data_com_pos.append(jnp.expand_dims(z_k['com_joint'][..., :3], axis=0)) # (1, 3)

            # Concatenate the com positions
            data_com_pos = jnp.concatenate(data_com_pos, axis=0) # (H, 3)
            return total_reward, data_com_pos


        action_samples = jnp.empty((1 + self.num_pi_traj + self.M, self.planning_horizon, self.act_dim))
        action_samples = action_samples.at[0:1].set(jnp.expand_dims(prev_mean, axis=0))
        action_samples = action_samples.at[1:self.num_pi_traj+1].set(warm_start)

        for _ in range(self.N):
            rng, rng_i = jax.random.split(rng)

            samples = jax.random.normal(rng_i, (self.M, self.planning_horizon, self.act_dim)) * std_dev + mean
            action_samples = action_samples.at[1+self.num_pi_traj:].set(samples)
            
            # Evaluate cost of the samples using vmap
            returns, data_com_pos = jax.vmap(total_reward)(action_samples) # (M + Pi + 1, 1), (M + Pi + 1, H, 3)

            # Select the top K% samples according to the returns
            elite_idxs = jnp.argsort(returns.squeeze(axis=-1), 
                            axis=-1)[-self.K:] # (K,)
            
            elite_set = action_samples[elite_idxs] # (K, horizon, nu)

            # MPPI step
            elite_returns = returns[elite_idxs] # (K, 1) # type: ignore
            
            max_ret = jnp.max(elite_returns)
            score = jnp.exp((elite_returns - max_ret) * self.temp)
            score = score / jnp.sum(score)
            score_expand = jnp.expand_dims(score, axis=-1) # (K, 1, 1)

            # (horizon, nu)
            _mean = jnp.sum(score_expand * elite_set, axis=0) / (score_expand.sum(axis=0) + 1e-6)

            _std_dev = jnp.sqrt(
                jnp.sum(score_expand * (elite_set - _mean)**2, axis=0) / (score_expand.sum(axis=0) + 1e-6)
            )
            _std_dev = jnp.clip(_std_dev, 0.005, 2.)


            # Fit a Gaussian to the elite set with momentum
            mean = (1 - self.smooth) * mean + self.smooth * _mean
            std_dev = (1 - self.smooth) * std_dev + self.smooth * _std_dev

        mpc_dist = distrax.MultivariateNormalDiag(loc=mean[0], scale_diag=std_dev[0])
        actions = mpc_dist.sample(seed=rng) # (nu,)

        return actions, mean, data_com_pos
