# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import copy
from typing import Dict
import torch
from torch import nn
import numpy as np

from . import tools
from . import networks

import torch.nn.functional as F

to_np = lambda x: x.detach().cpu().numpy()


class KinoDynamicComWorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera=True):
        super(KinoDynamicComWorldModel, self).__init__()

        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.device = self._config.device

        self._horizon_length = self._config.horizon

        self.encoder = networks.MultiEncoder(obs_shape, 
                                             **config.encoder, 
                                             use_camera=use_camera)
        self.embed_size = self.encoder.outdim

        self.dynamics = networks.KinoDynamicsCoMRSSM(
            config.task_cfg[config.task],
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            self.embed_size,
            config.device,
        )

        self.future_preds_dim = (self.dynamics._com_state_dim + self.dynamics._joint_state_dim) \
            * self._horizon_length

        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete \
                + config.dyn_deter \
                + self.dynamics._com_state_dim + self.dynamics._joint_state_dim
        else:
            feat_size = config.dyn_stoch + config.dyn_deter \
                + self.dynamics._com_state_dim + self.dynamics._joint_state_dim
        
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, obs_shape, **config.decoder, use_camera=use_camera
        )

        # TODO: Bug, add action?
        self.heads["reward"] = networks.MLP(
            feat_size + self.dynamics._actions_dim,
            {'reward': (1,)},
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            std=config.value_head["std"],
            min_std=config.value_head["min_std"],
            max_std=config.value_head["max_std"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        self.heads["value"] = networks.MLP(
            feat_size,
            {'value': (1,)},
            config.value_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.value_head["dist"],
            std=config.value_head["std"],
            min_std=config.value_head["min_std"],
            max_std=config.value_head["max_std"],
            outscale=config.value_head["outscale"],
            device=config.device,
            name="Value",
        )

        self.heads["actor"] = networks.MLP(
            feat_size,
            {'action': (self.dynamics._actions_dim,)},
            config.actor_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.actor_head["dist"],
            std=config.actor_head["std"],
            min_std=config.actor_head["min_std"],
            max_std=config.actor_head["max_std"],
            outscale=config.actor_head["outscale"],
            device=config.device,
            name="Actor",
        )

        # self.heads["cont"] = networks.MLP(
        #     feat_size,
        #     (),
        #     config.cont_head["layers"],
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist="binary",
        #     outscale=config.cont_head["outscale"],
        #     device=config.device,
        #     name="Cont",
        # )

        for name in config.grad_heads:
            assert name in self.heads, name
        
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            value=config.value_head["loss_scale"],
            actor=config.actor_head["loss_scale"],
            image=1.0,
            # cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # prop, image, reward, com_state, joint_state, action, value
        # Shape: (batch_size, traj_length, ...)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                com_joint_loss, com_joint_prior_value, com_joint_post_value = self.dynamics.com_joint_loss(post, prior, data)

                preds = {}
                for name, head in self.heads.items(): # decoder, reward, value, actor
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    if name == "reward":
                        pred = head(torch.cat([feat, data["action"]], dim=-1))
                    else:
                        pred = head(feat)

                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred

                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss + com_joint_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss) # Latent Dynamics Loss
        metrics["rep_loss"] = to_np(rep_loss) # Latent Representation Loss
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["estimation_loss"] = to_np(com_joint_post_value)
        metrics["dynamics_loss"] = to_np(com_joint_prior_value)
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, input_data: Dict):
        data = {}
        for key, value in input_data.items():
            if value is None:
                data[key] = None            
            elif isinstance(value, dict):
                data[key] = {k: v.to(self.device) for k, v in value.items()}
            else:
                data[key] = value.to(self.device)
        return data

    def forward(self, data):
        # Shape: (batch_size, ...)
        data = self.preprocess(data)
        with torch.no_grad():
            is_first = data["is_first"]
            wm_state = data['wm_state']

            obs = {"prop": data["prop"], "image": data["image"]}
            embed = self.encoder(obs)

            wm_state, _ = self.dynamics.check_reset(is_first, embed, wm_state)

            y = []
            for time in range(self._config.horizon):
                action_dist_dict = self.heads["actor"](self.dynamics.get_feat(wm_state))
                action = action_dist_dict['action'].sample()

                if time == 0:
                    wm_state_post, wm_state = self.dynamics.obs_step(
                            wm_state, action, embed, is_first    
                        )
                    wm_deter_feature_post = self.dynamics.get_deter_feat(wm_state_post)
                else:
                    wm_state = self.dynamics.img_step(wm_state, action)

                y.append(self.dynamics.get_com_joint_feat(wm_state))

            # concatenate all predictions
            y = torch.cat(y, dim=-1)

        return y, wm_deter_feature_post, wm_state_post

    def get_export_model(self):
        def distribution_exporter_for_jax(model, output_keys=("mean", "logits", "std")):
            """
            Wraps a PyTorch model so that its forward returns only mean arrays or std (not distributions).
            output_keys: which attributes to extract from the distribution object.
            """
            class JAXExportWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = copy.deepcopy(model)

                def forward(self, *args, **kwargs):
                    out = self.model(*args, **kwargs)
                    # If output is a dict of dists, extract arrays
                    if isinstance(out, dict):
                        new_out = {}
                        for k, v in out.items():
                            if hasattr(v, "mean"):
                                for key in output_keys:
                                    if hasattr(v, key):
                                        val = getattr(v, key)
                                        new_out[f"{k}_{key}"] = val() if callable(val) else val
                            else:
                                new_out[k] = v
                        return new_out
                    # If output is a dist, extract arrays
                    elif hasattr(out, "mean"):
                        return {key: getattr(out, key)() if callable(getattr(out, key)) else getattr(out, key)
                                for key in output_keys if hasattr(out, key)}
                    else:
                        return out

            return JAXExportWrapper(model)

        m = {}
        m["encoder"] = copy.deepcopy(self.encoder).eval()
        m["dynamics"] = copy.deepcopy(self.dynamics).eval()
        m["reward"] = distribution_exporter_for_jax(self.heads["reward"]).eval()
        m["value"] = distribution_exporter_for_jax(self.heads["value"]).eval()
        m["actor"] = distribution_exporter_for_jax(self.heads["actor"]).eval()
        m["decoder"] = distribution_exporter_for_jax(self.heads["decoder"]).eval()

        m = {key: value.requires_grad_(False).to(self.device) for key, value in m.items()}

        return m

# class KinoDynamicComWorldModel(nn.Module):
#     def __init__(self, config, obs_shape):
#         super(KinoDynamicComWorldModel, self).__init__()

#         self._use_amp = True if config.precision == 16 else False # TODO: implement this
#         self._config = config
#         self.device = self._config.device

#         self._prop_obs_dim : int = obs_shape['prop'][0]
#         self._image_obs_dim : int = np.array(obs_shape['image']).prod()
#         self._action_dim : int = 12

#         self._horizon_length = self._config.horizon

#         self._lambda = self._config.time_decay

#         self.future_preds_dim = (12 + 24) * (self._horizon_length + 1)

#         # self.polyak_counter = 0

#         # Network Definitions for world model.
#         self.heads = nn.ModuleDict()

#         self.heads['encoder'] = networks.DeterministicMultiEncoder(
#                 cnn_inp_shape=obs_shape['image'],
#                 mlp_inp_dim=self._prop_obs_dim,
#                 embed_dim=self._config.embed_dim,
                
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 act=self._config.act,
#             )

#         self.heads['dynamics'] = networks.KinoDynamicsCoMModel(
#                 embed_dim=self._config.embed_dim,

#                 latent_dim=self._config.latent_dim,
#                 latent_category_dim=self._config.latent_category_dim,
#                 hidden_state_dim=self._config.hidden_state_dim,

#                 action_dim=self._action_dim,
                
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 act=self._config.act,
#             )

#         self.heads['decoder'] = networks.DeterministicMultiDecoder(
#                 cnn_outdim=self.heads['encoder']._image_encoder.outdim,
#                 cnn_inp_shape=obs_shape['image'],
#                 mlp_inp_dim=self._prop_obs_dim,
#                 embed_dim=self.heads['dynamics']._output_dim,
                
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 act=self._config.act,
#             )       
        
#         self.heads['rewards'] = networks.DeterministicMLP(
#                 input_dim=self.heads['dynamics']._output_dim,
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 output_dim=1,
#                 act=self._config.act,
#             )

#         self.heads['value'] = networks.DeterministicMLP(
#                 input_dim=self.heads['dynamics']._output_dim,
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 output_dim=1,
#                 act=self._config.act,
#             )

#         # Policy
#         self.heads['policy'] = networks.DeterministicMLP(
#             input_dim=self.heads['dynamics']._output_dim,
#             hidden_layer_sizes=self._config.hidden_sizes,
#             output_dim=self._action_dim,
#             act=self._config.act,
#         )
        
#         self._model_opt = torch.optim.Adam(self.parameters(), lr=config.model_lr)

#         print(
#                 f"World Model Optimizer has {sum(param.numel() for param in self.parameters())} variables."
#             )

#     def _train(self, data):
#         data = self.preprocess(data)
#         # prop, image, reward, com_state, joint_state, action, value
#         # Shape: (batch_size, traj_length, ...)

#         com_pos_ref = data["com_state"][:, 0, :2].clone()
#         batch_size = data["prop"].shape[0]
#         traj_length = data["prop"].shape[1]
#         device = data["prop"].device

#         self.train() # Enable train mode

#         # Kinodynamics losses
#         loss_actions = 0.
#         loss_dynamics = 0.
#         loss_rewards = 0.
#         loss_value = 0.
#         loss_consistency = 0.
#         loss_reconstruction = 0.
#         loss_estimator = 0.

#         embed = self.heads['encoder'].train_forward(data["prop"], data["image"])

#         for i in range(0, traj_length-1):
#             # TODO: add is_first to handle resets in trajectory
#             if i == 0:
#                 states_i = self.heads['dynamics'].initial(batch_size, device, embed[:, i])
#             else:
#                 states_i = states_ip1

#             svec_i = torch.cat([states_i['prior']['com_state'], 
#                                 states_i['prior']['joint_state'],
#                                 states_i['prior']['latent_sample'],
#                                 states_i['hidden_state']
#                             ], dim=-1)

#             # reward prediction
#             reward_i = self.heads['rewards'](svec_i)
#             loss_rewards += F.mse_loss(reward_i, data["reward"][:, i, ...].clone()) * (self._lambda ** i)

#             # value prediction.
#             value_i = self.heads['value'](svec_i)
#             loss_value += F.mse_loss(value_i, data["value"][:, i, ...].clone()) * (self._lambda ** i)

#             # dreamer action prediction
#             dreamer_action_i = self.heads['policy'](svec_i.detach())
#             loss_actions += F.mse_loss(dreamer_action_i, data["action"][:, i, ...].clone()) * (self._lambda ** i)


#             # future state prediction
#             states_ip1 = self.heads['dynamics'].train_forward(
#                     states_i['prior']['com_state'], states_i['prior']['joint_state'],
#                     states_i['prior']['latent_sample'], data["action"][:, i, ...].clone(), 
#                     states_i['hidden_state'], embed[:, i+1]
#                 )

#             # because we don't have global positioning system on hw, predict com with respect to the first frame
#             com_state_ip1_target = data["com_state"][:, i+1, ...].clone()
#             com_state_ip1_target[..., :2] -= com_pos_ref
#             loss_dynamics += (F.mse_loss(states_ip1['prior']['com_state'], com_state_ip1_target) + \
#                                 F.mse_loss(states_ip1['prior']['joint_state'], data["joint_state"][:, i+1, ...].clone())
#                             ) * (self._lambda ** i)

#             com_state_ip1_esti_target = data["com_state"][:, i+1, ...].clone()
#             com_state_ip1_esti_target[..., :2] = 0.
#             loss_estimator += (F.mse_loss(states_ip1['post']['com_state'], com_state_ip1_esti_target) + \
#                                 F.mse_loss(states_ip1['post']['joint_state'], data["joint_state"][:, i+1, ...].clone())
#                             ) * (self._lambda ** i)


#             # Reconstruction & KL
#             svec_ip1 = torch.cat([states_ip1['post']['com_state'],
#                                   states_ip1['post']['joint_state'],
#                                   states_ip1['post']['latent_sample'],
#                                   states_ip1['hidden_state']
#                             ], dim=-1)
#             # # reconstruction
#             recon_prop_ip1, recon_image_ip1 = self.heads['decoder'](svec_ip1)
#             loss_reconstruction += (F.mse_loss(recon_prop_ip1, data["prop"][:, i+1, ...].clone()) + \
#                                         F.mse_loss(recon_image_ip1, data['image'][:, i+1, ...].clone())
#                                     ) * (self._lambda ** i)

#             l_rep = F.kl_div(
#                         input=F.log_softmax(states_ip1['post']['latent_logits'], dim=-1),
#                         target=F.softmax(states_ip1['prior']['latent_logits'].detach(), dim=-1),
#                         reduction='batchmean')
#             l_dyn = F.kl_div(
#                         input=F.log_softmax(states_ip1['post']['latent_logits'].detach(), dim=-1),
#                         target=F.softmax(states_ip1['prior']['latent_logits'], dim=-1),
#                         reduction='batchmean')

#             loss_consistency += (torch.clip(l_rep, min=self._config.scale_kl_clip) * self._config.scale_kl_dyn
#                                 + torch.clip(l_dyn, min=self._config.scale_kl_clip) * self._config.scale_kl_rep
#                                 ) * (self._lambda ** i)

#         loss_model = torch.mean(
#                                 self._config.scale_rew * loss_rewards.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_val * loss_value.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_act * loss_actions.clamp(max=1e3) + # type: ignore

#                                 self._config.scale_dyn * loss_dynamics.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_est * loss_estimator.clamp(max=1e3) + # type: ignore

#                                 self._config.scale_rec * loss_reconstruction.clamp(max=1e3) + # type: ignore

#                                 self._config.scale_kl * loss_consistency.clamp(max=1e3) # type: ignore
#                                 )

#         loss_model.register_hook(lambda grad: grad * (1/traj_length))

#         self._model_opt.zero_grad()
#         loss_model.backward()
#         norm = nn.utils.clip_grad_norm_(self.parameters(), self._config.grad_clip)
#         self._model_opt.step()

#         metrics = {'model_grad_norm': to_np(norm).item()}
#         losses = {
#             "actions": loss_actions,
#             "rewards": loss_rewards,
#             "dynamics": loss_dynamics,
#             "value": loss_value,
#             "consistency": loss_consistency,
#             "reconstruction": loss_reconstruction,
#             "estimator": loss_estimator,
#         }
#         metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})

#         self.eval() # Enable eval mode


#         # # Polyak update target network
#         # self.polyak_counter += 1
#         # if self.polyak_counter % self._config.polyak_freq == 0:
#         #     with torch.no_grad():
#         #         for p, p_target in zip(self.heads['estimator'].parameters(), self.heads['estimator'].parameters()):
#         #             p_target.data.lerp_(p.data, self._config.polyak_tau)
#         #     self.polyak_counter = 0

#         return metrics

#     # this function is called during both rollout and training
#     def preprocess(self, input_data: Dict):
#         data = {}
#         for key, value in input_data.items():
#             if value is not None:
#                 data[key] = value.to(self.device)
#             else:
#                 data[key] = None
#         return data

#     def forward(self, data):
#         data = self.preprocess(data)
#         # Shape: (batch_size, ...)
#         with torch.no_grad():
#             is_first = data["is_first"]
#             batch_size = data["prop"].shape[0]
#             device = data["prop"].device

#             embed = self.heads['encoder'](data["prop"], data["image"])

#             if data["hidden_state"] == None:
#                 state_t = self.heads['dynamics'].initial(batch_size, device, embed)
#             else:
#                 state_t = self.heads['dynamics'].initial_mask(embed, data["hidden_state"], (is_first == 1))                

#             y = []
#             y.append(
#                     torch.cat(
#                         [state_t['prior']['com_state'], 
#                          state_t['prior']['joint_state']], dim=-1)
#                 )


#             for time_step in range(self._horizon_length):
#                 # dreamer action prediction
#                 dreamer_action_t = self.heads['policy'](
#                         torch.cat(
#                             [state_t['prior']['com_state'],
#                                 state_t['prior']['joint_state'], 
#                                 state_t['prior']['latent_sample'], 
#                                 state_t['hidden_state']
#                             ], dim=-1)
#                     )

#                 # future state prediction
#                 states_tp1 = self.heads['dynamics'](
#                         state_t['prior']['com_state'],
#                         state_t['prior']['joint_state'],
#                         state_t['prior']['latent_sample'],
#                         dreamer_action_t,
#                         state_t['hidden_state'],
#                     )

#                 if time_step == 0:
#                     feature = states_tp1['hidden_state'].clone()

#                 # append future predictions 
#                 y.append(
#                         torch.cat(
#                             [states_tp1['prior']['com_state'], 
#                              states_tp1['prior']['joint_state']], dim=-1)
#                     )

#                 # update states
#                 state_t = states_tp1

#             # concatenate all predictions
#             y = torch.cat(y, dim=-1)

#         return y, feature


# class KinoDynamicComWorldModel(nn.Module):
#     def __init__(self, config, obs_shape):
#         super(KinoDynamicComWorldModel, self).__init__()

#         self._use_amp = True if config.precision == 16 else False # TODO: implement this
#         self._config = config
#         self.device = self._config.device

#         self._prop_obs_dim : int = obs_shape['prop'][0]
#         self._image_obs_dim : int = np.array(obs_shape['image']).prod()
#         self._action_dim : int = 12

#         self._horizon_length = self._config.horizon

#         self._lambda = self._config.time_decay

#         self.future_preds_dim = (12 + 24) * (self._horizon_length + 1)
#         # self.future_preds_dim = (12 + 24 + 
#         #     self._config.latent_dim * self._config.latent_category_dim) \
#         #     * (self._horizon_length + 1)

#         self.polyak_counter = 0

#         # Network Definitions for world model.
#         self.heads = nn.ModuleDict()

#         self.heads['dynamics'] = networks.KinoDynamicsCoMModel(
#                 latent_dim=self._config.latent_dim,
#                 latent_category_dim=self._config.latent_category_dim,
#                 hidden_state_dim=self._config.hidden_state_dim,

#                 action_dim=self._action_dim,
                
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 act=self._config.act,
#             )


#         self.heads['estimator'] = networks.KinoDynamicCoMEstimator(
#                 cnn_inp_shape=obs_shape['image'],
#                 mlp_inp_dim=self._prop_obs_dim,
#                 prop_out_dim=self._config.estimator_mlp_out_dim,

#                 latent_dim=self._config.latent_dim,
#                 latent_category_dim=self._config.latent_category_dim,

#                 com_state_dim=self.heads['dynamics']._com_state_dim,
#                 joint_state_dim=self.heads['dynamics']._joint_state_dim,

#                 hidden_layer_sizes=self._config.estimator_hidden_sizes,
#                 act=self._config.estimator_act,
#             )
#         # self.heads['estimator'] = copy.deepcopy(self.heads['estimator'])


#         self.heads['decoder'] = networks.DeterministicMultiDecoder(
#                 input_dim=self.heads['dynamics']._output_dim,
#                 mlp_feature_dim=self._config.decoder_mlp_out_dim,
#                 cnn_feature_dim=self._config.decoder_cnn_out_dim,
#                 cnn_out_shape=obs_shape['image'],
#                 prop_out_dim=self._prop_obs_dim,
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 act=self._config.act
#             )


#         self.heads['rewards'] = networks.DeterministicMLP(
#                 input_dim=self.heads['dynamics']._output_dim,
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 output_dim=1,
#                 act=self._config.act,
#             )


#         self.heads['value'] = networks.DeterministicMLP(
#                 input_dim=self.heads['dynamics']._output_dim,
#                 hidden_layer_sizes=self._config.hidden_sizes,
#                 output_dim=1,
#                 act=self._config.act,
#             )


#         # Policy
#         self.heads['policy'] = networks.DeterministicMLP(
#             input_dim=self.heads['dynamics']._output_dim,
#             hidden_layer_sizes=self._config.hidden_sizes,
#             output_dim=self._action_dim,
#             act=self._config.act,
#         )
        
#         self._model_opt = torch.optim.Adam(self.parameters(), lr=config.model_lr)

#         print(
#                 f"World Model Optimizer has {sum(param.numel() for param in self.parameters())} variables."
#             )

#     def _train(self, data):
#         data = self.preprocess(data)
#         # prop, image, reward, com_state, joint_state, action, value
#         # Shape: (batch_size, traj_length, ...)

#         com_pos_ref = data["com_state"][:, 0, :2].clone()
#         batch_size = data["prop"].shape[0]
#         traj_length = data["prop"].shape[1]
#         device = data["prop"].device

#         self.train() # Enable train mode

#         # Kinodynamics losses
#         loss_actions = 0.
#         loss_dynamics = 0.
#         loss_rewards = 0.
#         loss_value = 0.
#         loss_consistency = 0.
#         loss_reconstruction = 0.
#         loss_estimator = 0.

#         # TODO: add is_first to handle resets in trajectory
#         for i in range(0, traj_length-1):
#             obs_i = {}
#             obs_i["prop"] = data["prop"][:, i, ...].clone()
#             obs_i["image"] = data["image"][:, i, ...].clone()
            

#             if i == 0:
#                 estimates_i = self.heads['estimator'](obs_i) # z_t
#                 states_i = estimates_i
#                 states_i['hidden_state'] = self.heads['dynamics'].initial(batch_size, device)

#             else:
#                 estimates_i = estimates_ip1
#                 states_i = states_ip1


#             # because we don't have global positioning system on hw, predict com with respect to the first frame
#             com_state_esti_target = data["com_state"][:, i, ...].clone()
#             com_state_esti_target[..., :2] -= com_pos_ref
#             loss_estimator += (F.mse_loss(estimates_i['com_state'], com_state_esti_target) + \
#                 F.mse_loss(estimates_i['joint_state'], data["joint_state"][:, i, ...].clone())) * (self._lambda ** i) / traj_length


#             svec_i = torch.cat([states_i['com_state'], 
#                                 states_i['joint_state'],
#                                 states_i['latent_sample'],
#                                 states_i['hidden_state']
#                             ], dim=-1)

#             # reward prediction
#             reward_i = self.heads['rewards'](svec_i.detach())
#             loss_rewards += F.mse_loss(reward_i, data["reward"][:, i, ...].clone()) * (self._lambda ** i) / traj_length

#             # value prediction.
#             value_i = self.heads['value'](svec_i.detach())
#             loss_value += F.mse_loss(value_i, data["value"][:, i, ...].clone()) * (self._lambda ** i) / traj_length


#             # future state prediction
#             states_ip1 = self.heads['dynamics'](
#                     states_i['com_state'], states_i['joint_state'], states_i['latent_sample'], 
#                     data["action"][:, i, ...].clone(), states_i['hidden_state']
#                 )

#             com_state_ip1_target = data["com_state"][:, i+1, ...].clone()
#             com_state_ip1_target[..., :2] -= com_pos_ref
#             loss_dynamics += (F.mse_loss(states_ip1['com_state'], com_state_ip1_target) + \
#                                 F.mse_loss(states_ip1['joint_state'], data["joint_state"][:, i+1, ...].clone())) * (self._lambda ** i) / traj_length


#             # Reconstruction & KL
#             svec_ip1 = torch.cat([states_ip1['com_state'],
#                                   states_ip1['joint_state'],
#                                   states_ip1['latent_sample'],
#                                   states_ip1['hidden_state']
#                             ], dim=-1)

#             with torch.no_grad():
#                 obs_ip1 = {}
#                 obs_ip1["prop"] = data["prop"][:, i+1, ...].clone()
#                 obs_ip1["image"] = data["image"][:, i+1, ...].clone()

#             estimates_ip1 = self.heads['estimator'](obs_ip1)

#             # reconstruction
#             recon_obs_ip1 = self.heads['decoder'](svec_ip1)
#             loss_reconstruction += (F.mse_loss(recon_obs_ip1['prop'], obs_ip1["prop"]) + \
#                                     F.mse_loss(recon_obs_ip1['image'], obs_ip1['image'])) * (self._lambda ** i) / traj_length

#             # KL(latent_logits_ip1 || latent_logits_ip1_tgt)
#             loss_consistency += F.kl_div(
#                     input=F.log_softmax(estimates_ip1['latent_logits'], dim=-1),
#                     target=F.softmax(states_ip1['latent_logits'], dim=-1),
#                     reduction='batchmean'
#                 ) * (self._lambda ** i) / traj_length


#             # dreamer action prediction
#             dreamer_action_i = self.heads['policy'](svec_i.detach())
#             loss_actions += F.mse_loss(dreamer_action_i, data["action"][:, i, ...].clone()) * (self._lambda ** i) / traj_length

#         loss_model = torch.mean(
#                                 self._config.scale_act * loss_actions.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_rew * loss_rewards.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_dyn * loss_dynamics.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_val * loss_value.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_est * loss_estimator.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_rec * loss_reconstruction.clamp(max=1e3) + # type: ignore
#                                 self._config.scale_kl * loss_consistency.clamp(max=1e3) # type: ignore
#                                 )

#         # loss_model.register_hook(lambda grad: grad * (1/traj_length))

#         self._model_opt.zero_grad()
#         loss_model.backward()
#         norm = nn.utils.clip_grad_norm_(self.parameters(), self._config.grad_clip)
#         self._model_opt.step()

#         metrics = {'model_grad_norm': to_np(norm).item()}
#         losses = {
#             "actions": loss_actions,
#             "rewards": loss_rewards,
#             "dynamics": loss_dynamics,
#             "value": loss_value,
#             "consistency": loss_consistency,
#             "reconstruction": loss_reconstruction,
#             "estimator": loss_estimator,
#         }
#         metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})

#         self.eval() # Enable eval mode


#         # # Polyak update target network
#         # self.polyak_counter += 1
#         # if self.polyak_counter % self._config.polyak_freq == 0:
#         #     with torch.no_grad():
#         #         for p, p_target in zip(self.heads['estimator'].parameters(), self.heads['estimator'].parameters()):
#         #             p_target.data.lerp_(p.data, self._config.polyak_tau)
#         #     self.polyak_counter = 0

#         return metrics

#     # this function is called during both rollout and training
#     def preprocess(self, input_data: Dict):
#         data = {}
#         for key, value in input_data.items():
#             if value is not None:
#                 data[key] = value.to(self.device)
#             else:
#                 data[key] = None
#         return data

#     def forward(self, data):
#         data = self.preprocess(data)
#         # Shape: (batch_size, ...)
#         with torch.no_grad():
#             is_first = data["is_first"]
#             batch_size = data["prop"].shape[0]
#             device = data["prop"].device


#             obs = {"prop": data["prop"], "image": data["image"]}

#             # estimate com_state, joint_state, latent_sample
#             estimates_0 = self.heads['estimator'](obs)

#             com_state_t = estimates_0['com_state']
#             joint_state_t = estimates_0['joint_state']

#             latent_sample_t = estimates_0['latent_sample']

#             hidden_state_t = data["hidden_state"]
#             if hidden_state_t == None:
#                 hidden_state_t = self.heads['dynamics'].initial(batch_size, device)
#             else:
#                 # if is_first then reset hidden state for that environment
#                 hidden_state_t[is_first == 1] = self.heads['dynamics'].initial(batch_size, device)[is_first == 1]

#             y = []
#             y.append(
#                     torch.cat(
#                         [com_state_t, joint_state_t], dim=-1)
#                 )


#             for time_step in range(self._horizon_length):
#                 # dreamer action prediction
#                 dreamer_action_t = self.heads['policy'](
#                         torch.cat(
#                             [com_state_t, joint_state_t, latent_sample_t, hidden_state_t], dim=-1)
#                     )
                
#                 # future state prediction
#                 preds_future_tp1 = self.heads['dynamics'](
#                         com_state_t, joint_state_t, latent_sample_t, dreamer_action_t, hidden_state_t
#                     )

#                 if time_step == 0:
#                     feature = preds_future_tp1['hidden_state'].clone()

#                 # append future predictions 
#                 y.append(
#                         torch.cat(
#                             [preds_future_tp1['com_state'], 
#                              preds_future_tp1['joint_state']], dim=-1)
#                     )

#                 # update states
#                 com_state_t = preds_future_tp1['com_state']
#                 joint_state_t = preds_future_tp1['joint_state']
#                 latent_sample_t = preds_future_tp1['latent_sample']
#                 hidden_state_t = preds_future_tp1['hidden_state']

#             # concatenate all predictions
#             y = torch.cat(y, dim=-1)

#         return y, feature

