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

import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

from . import tools


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_deter_feat(self, state):
        return state["deter"]

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)

        # p_theta
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)

        # GRU
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)

        # q_theta
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        use_camera = False,
    ):
        super(MultiEncoder, self).__init__()
        self.use_camera = use_camera
        excluded = ("is_first", "is_last", "is_terminal", "reward",  "height_map")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = TimeConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
            print('cnn outdim', self._cnn.outdim)
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units
            print('mlp outdim', mlp_units)

        print('total outdim:', self.outdim)

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            if(self.use_camera):
                inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
                outputs.append(self._cnn(inputs))
            else:
                outputs.append(torch.zeros((obs["is_first"].shape + (self._cnn.outdim,)), device=obs["is_first"].device))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
        use_camera=False,
    ):
        super(MultiDecoder, self).__init__()
        self.use_camera = use_camera
        excluded = ("is_first", "is_last", "is_terminal", "height_map")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = TimeConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes and self.use_camera:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class TimeConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(TimeConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h+1) // 2, (w+1) // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch) [or] ignore if time is not present
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # print('init encoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class TimeConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act='ELU',
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        # add this to fully recover the process of conv encoder
        input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h+1) // 2, (w+1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])
        self.w_list.append(shape[2])

        super(TimeConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1
        # layer_num = int(np.log2(shape[2]) - np.log2(minres))
        # self._minres = minres
        # out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2

        layers = []
        # h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            # pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            # pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)

            if(self.h_list[i] * 2 == self.h_list[i+1]):
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if(self.w_list[i] * 2 == self.w_list[i+1]):
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            # h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)  [or] ignore if time is not present
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])]
        )
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # print('init decoder shape:', x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        x = self.layers(x)
        # (batch * time, -1) -> (batch, time, ch, h, w) [or] ignore if time is not present
        mean = x.reshape(features.shape[:-1] + self._shape)

        if len(mean.shape) == 5:
            # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
            mean = mean.permute(0, 1, 3, 4, 2)
        else:
            # (batch, ch, h, w) -> (batch, h, w, ch)
            mean = mean.permute(0, 2, 3, 1)

        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        # else:
        #     mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DeterministicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim,  act="SiLU", norm=False):
        super(DeterministicMLP, self).__init__()
        act = getattr(torch.nn, act)
        self.num_hidden_dim = len(hidden_layer_sizes)
        self.layers = nn.Sequential()
        
        self.layers.add_module(
            "input_linear", nn.Linear(input_dim, hidden_layer_sizes[0], bias=False)
        )
        if norm:
            self.layers.add_module(
                "input_norm", nn.LayerNorm(hidden_layer_sizes[0], eps=1e-03)
            )
        self.layers.add_module("input_act", act())

        for i in range(self.num_hidden_dim - 1):
            self.layers.add_module(
                f"hidden_linear{i}", nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1], bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"hidden_norm{i}", nn.LayerNorm(hidden_layer_sizes[i + 1], eps=1e-03)
                )
            self.layers.add_module(f"hidden_act{i}", act())
        
        self.layers.add_module(
            "output_linear", nn.Linear(hidden_layer_sizes[-1], output_dim, bias=False)
        )

        self.layers.apply(tools.weight_init) 
        
    def forward(self, x):
        x = self.layers(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h+1) // 2, (w+1) // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # (batch, h, w, ch) -> (batch, ch, h, w)
        x = obs.permute(0, 3, 1, 2)

        x = self.layers(x) # (batch, ch, h, w)

        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(64, 64, 1),
        depth=32,
        act='ELU',
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        # add this to fully recover the process of conv encoder
        h, w, input_ch = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for i in range(stages):
            h, w = (h+1) // 2, (w+1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[0])
        self.w_list.append(shape[1])

        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1

        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2

        layers = []
        # h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[2]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            # pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            # pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)

            if(self.h_list[i] * 2 == self.h_list[i+1]):
                pad_h, outpad_h = 1, 0
            else:
                pad_h, outpad_h = 2, 1

            if(self.w_list[i] * 2 == self.w_list[i+1]):
                pad_w, outpad_w = 1, 0
            else:
                pad_w, outpad_w = 2, 1

            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            # h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features):
        x = self._linear_layer(features)
        # (batch, -1) -> (batch, h, w, ch)
        x = x.reshape(
            [-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])]
        )

        # (batch, h, w, ch) -> (batch, ch, h, w)
        x = x.permute(0, 3, 1, 2)

        x = self.layers(x) # (batch, ch, h, w)

        # (batch, ch, h, w) -> (batch, h, w, ch)
        image = x.permute(0, 2, 3, 1)
        image += 0.5

        return image


class DeterministicMultiEncoder(nn.Module): 
    def __init__(
        self,
        cnn_inp_shape, mlp_inp_dim, 
        embed_dim,
        hidden_layer_sizes,
        act="Tanh", norm=False,
    ):
        super(DeterministicMultiEncoder, self).__init__()

        self._image_encoder = ConvEncoder(
                input_shape=cnn_inp_shape,
                act=act,
                norm=norm,
            )

        mlp_out_dim = 4 * embed_dim
        self._mlp_encoder = DeterministicMLP(
                input_dim=mlp_inp_dim,
                output_dim=mlp_out_dim,
                act=act,
                norm=norm,
                hidden_layer_sizes=hidden_layer_sizes,
            )
        
        self._mixer_encoder = DeterministicMLP(
                input_dim=mlp_out_dim + self._image_encoder.outdim,
                output_dim=embed_dim,
                act=act,
                norm=norm,
                hidden_layer_sizes=hidden_layer_sizes,
            )

    def forward(self, x, image):
        # image encoder
        image_embedding = self._image_encoder(image)

        # mlp encoder
        x_embedding = self._mlp_encoder(x)

        # mixer encoder
        return self._mixer_encoder(torch.cat([x_embedding, image_embedding], dim=-1))

    def train_forward(self, x, image):
        """
        x: (batch, t, mlp_inp_dim)
        image: (batch, t, h, w, ch)
        """
        batch_dim = x.shape[0]
        time_dim = x.shape[1]
        image_shape = image.shape[2:]

        # x: (batch * t, mlp_inp_dim)
        x = x.reshape(batch_dim * time_dim, -1)
        # image: (batch * t, h, w, ch)
        image = image.reshape(batch_dim * time_dim, *image_shape)
 
        # encode
        embedding = self.forward(x, image)

        # embedding: (batch, t, embed_dim)
        embedding = embedding.reshape(batch_dim, time_dim, -1)

        return embedding


class DeterministicMultiDecoder(nn.Module):
    def __init__(
        self,
        cnn_outdim,
        cnn_inp_shape, mlp_inp_dim, 
        embed_dim,
        hidden_layer_sizes,
        act="Tanh", norm=False,
    ):
        super(DeterministicMultiDecoder, self).__init__()
        self.cnn_outdim = cnn_outdim
        self._image_decoder = ConvDecoder(
                feat_size=cnn_outdim,
                shape=cnn_inp_shape,
                act=act,
                norm=norm,
            )

        mlp_out_dim = 4 * embed_dim
        self._mlp_decoder = DeterministicMLP(
                input_dim=mlp_out_dim,
                output_dim=mlp_inp_dim,
                act=act,
                norm=norm,
                hidden_layer_sizes=hidden_layer_sizes,
            )
        
        self._mixer_decoder = DeterministicMLP(
                input_dim=embed_dim,
                output_dim=cnn_outdim + mlp_out_dim,
                act=act,
                norm=norm,
                hidden_layer_sizes=hidden_layer_sizes,
            )

    def forward(self, embedding):
        # mixer decoder
        x_embedding = self._mixer_decoder(embedding)

        # mlp decoder
        x = self._mlp_decoder(x_embedding[..., self.cnn_outdim:])

        # image decoder
        image = self._image_decoder(x_embedding[..., :self.cnn_outdim])

        return x, image

    def train_forward(self, embedding):
        """
        x: (batch, t, mlp_inp_dim)
        image: (batch, t, h, w, ch)
        """
        # embedding: (batch, t, embed_dim)
        batch_dim = embedding.shape[0]
        time_dim = embedding.shape[1]

        # decode
        x_recon, image_recon = self.forward(embedding.reshape(batch_dim * time_dim, -1))

        # x_recon: (batch, t, mlp_inp_dim)
        x_recon = x_recon.reshape(batch_dim, time_dim, -1)
        # image_recon: (batch, t, h, w, ch)
        image_recon = image_recon.reshape(batch_dim, time_dim, *self._image_decoder._shape)

        return x_recon, image_recon


class KinoDynamicsCoMModel(nn.Module):
    def __init__(
            self,
            embed_dim,

            latent_dim,
            latent_category_dim,
            hidden_state_dim,
            
            action_dim,
            
            hidden_layer_sizes=[128, 64],
            act="Tanh",
        ):
        super(KinoDynamicsCoMModel, self).__init__()

        self._embed_dim: int = embed_dim

        self._action_dim: int = action_dim

        # Constants
        self._com_state_dim : int = 3 + 3 + 3 + 3 # p + rpy + lin_vel + ang_vel
        self._joint_state_dim: int = 12 + 12 # joint_pos + joint_vel
        self._action_transform_dim: int = 6 + 12 # force + torque + joint_vel_delta

        self._latent_dim: int = latent_dim
        self._latent_category_dim: int = latent_category_dim
        self._latent_logits_dim: int = latent_dim * latent_category_dim

        self._hidden_state_dim: int = hidden_state_dim

        self.gravity = [0, 0, -9.81] # m/s^2

        self.dt = 0.01 # time step.

        self._gru = StatelessGRUCell(
                input_size=self._com_state_dim + self._joint_state_dim \
                    + self._latent_logits_dim + action_dim,
                hidden_state_size=hidden_state_dim
            )
        
        # Dynamics Predictor (Prior): z_hat ~ p_theta(. | h)
        self._p_theta = DeterministicMLP(
                input_dim=hidden_state_dim,
                hidden_layer_sizes=hidden_layer_sizes,
                output_dim=self._action_transform_dim \
                    + self._latent_logits_dim,
                act=act
            )
        
        # Encoder (Post): z ~ q_theta(. | h, embed)
        self._q_theta = DeterministicMLP(
                input_dim=hidden_state_dim + embed_dim,
                hidden_layer_sizes=hidden_layer_sizes,
                output_dim=self._com_state_dim + self._joint_state_dim \
                    + self._latent_logits_dim,
                act=act
            )

        self._com_mass_sqrt = nn.Parameter(torch.ones(1))
        self._com_intertia_sqrt = nn.Parameter(torch.ones(3))

        self._output_dim = self._com_state_dim + self._joint_state_dim \
            + self._latent_logits_dim + self._hidden_state_dim

    def initial(self, batch_size, device, embed):
        hidden = torch.zeros(batch_size, self._hidden_state_dim).to(device)

        post_dict = self.post_estimate(hidden, embed)

        return {'prior': post_dict['post'], # make prior same as post
                'hidden_state': post_dict['hidden_state']}

    def initial_mask(self, embed, hidden, mask):
        hidden[mask] = torch.zeros_like(hidden[mask])
        post_dict = self.post_estimate(hidden, embed)

        return {'prior': post_dict['post'], # make prior same as post
                'hidden_state': post_dict['hidden_state']}

    def post_estimate(self, hidden_state, embed):
        post_output = self._q_theta(
                torch.cat([hidden_state, embed], dim=-1)
            )
        post_com_state = post_output[..., :self._com_state_dim]
        post_joint_state = post_output[..., self._com_state_dim:self._com_state_dim + self._joint_state_dim]
        post_latent_logits = post_output[..., self._com_state_dim + self._joint_state_dim:]

        post_latent_sample = self.logits_to_sample(post_latent_logits)
        post = {
                "com_state": post_com_state,
                "joint_state": post_joint_state,
                "latent_logits": post_latent_logits,
                "latent_sample": post_latent_sample,
            }

        return {"post": post, "hidden_state": hidden_state}

    def logits_to_sample(self, latent_logits):
        latent_logits = latent_logits.view(-1, self._latent_dim, self._latent_category_dim)
        latent_sample = F.gumbel_softmax(
                latent_logits,
                tau=1, hard=True, dim=-1
            ).view(-1, self._latent_dim * self._latent_category_dim)
        return latent_sample

    def train_forward(self, com_state, joint_state, latent_sample, action, hidden_state, embed):
        prior_dict = self.forward(com_state, joint_state, latent_sample, action, hidden_state)

        # Posterior Inference
        post_dict = self.post_estimate(prior_dict["hidden_state"], embed)

        # Add two dictionaries
        return {
                "prior": prior_dict["prior"],
                "post": post_dict["post"],
                "hidden_state": post_dict["hidden_state"],
            }

    def forward(self, com_state, joint_state, latent_sample, action, hidden_state):
        # Just Dynamics Prediction (Prior)
        new_hidden_state = self._gru(
                torch.cat([com_state, joint_state, latent_sample, action], dim=-1),
                hidden_state
            )
        prior_output = self._p_theta(new_hidden_state)
        prior_action_transform = prior_output[..., :self._action_transform_dim]
        prior_latent_logits = prior_output[..., self._action_transform_dim:]
        prior_latent_sample = self.logits_to_sample(prior_latent_logits)
        prior_com_state, prior_joint_state = self.eom(com_state, joint_state, prior_action_transform)
        prior = {
                "com_state": prior_com_state,
                "joint_state": prior_joint_state,
                "latent_logits": prior_latent_logits,
                "latent_sample": prior_latent_sample,
            }
        return {"prior": prior, "hidden_state": new_hidden_state}

    def eom(self, com_state_curr, joint_state, action_transform):
       # equations of motions.
       # TODO: wrap around joint limits & [-pi, pi].
       
       mass = torch.square(self._com_mass_sqrt) + 1e-6
       inertia = torch.square(self._com_intertia_sqrt) + 1e-6
       
       # logic to split com_state_curr into p, theta, lin_vel, ang_vel.
       p = com_state_curr[..., :3]
       theta = com_state_curr[..., 3:6]
       lin_vel = com_state_curr[..., 6:9]
       ang_vel = com_state_curr[..., 9:12]
       
       # logic to split action_transform into force, torque, joint_vel_delta.
       force = action_transform[..., :3]
       torque = action_transform[..., 3:6]
       joint_vel_delta = action_transform[..., 6:18]
       
       #logic to split joint_state into joint_pos, joint_vel.
       joint_pos = joint_state[..., :12]
       joint_vel = joint_state[..., 12:24]

       # CoM Dynamics       
       # continuous dynamics.
       ang_acc = (1/inertia) * (-tools.cross(ang_vel, inertia * ang_vel) + torque)
       lin_acc = (1/mass) * force + torch.tensor(self.gravity).to(com_state_curr.device)
       
       # integrate dynamics.
       new_ang_vel = ang_vel + ang_acc * self.dt
       new_lin_vel = lin_vel + lin_acc * self.dt
    
       new_theta_dot = tools.angvel_to_rpydot(theta, ang_vel)
       new_theta = theta + new_theta_dot * self.dt
       new_p = p + lin_vel * self.dt

       new_com_state = torch.cat([new_p, new_theta, new_lin_vel, new_ang_vel], dim=-1)

       # Kinematics
       new_joint_vel = joint_vel + joint_vel_delta
       new_joint_pos = joint_pos + new_joint_vel * self.dt
       new_joint_state = torch.cat([new_joint_pos, new_joint_vel], dim=-1)

       return new_com_state, new_joint_state


class StatelessGRUCell(nn.Module):
    def __init__(self, input_size, hidden_state_size):
        super(StatelessGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        # Update gate parameters (z_t)
        self.W_z = nn.Linear(input_size, hidden_state_size, bias=True)
        self.U_z = nn.Linear(hidden_state_size, hidden_state_size, bias=False)

        # Reset gate parameters (r_t)
        self.W_r = nn.Linear(input_size, hidden_state_size, bias=True)
        self.U_r = nn.Linear(hidden_state_size, hidden_state_size, bias=False)

        # Candidate hidden state parameters (\tilde{h}_t)
        self.W_h = nn.Linear(input_size, hidden_state_size, bias=True)
        self.U_h = nn.Linear(hidden_state_size, hidden_state_size, bias=False)

    def forward(self, x_t, h_prev):
        """
        Forward pass for the GRU cell.
        Args:
            x_t: Input tensor at time step t, shape (batch_size, input_size)
            h_prev: Previous hidden state, shape (batch_size, hidden_size)
        Returns:
            h_t: Next hidden state, shape (batch_size, hidden_size)
        """
        # Compute update gate (z_t)
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))

        # Compute reset gate (r_t)
        r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_prev))

        # Compute candidate hidden state (\tilde{h}_t)
        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_prev))

        # Compute next hidden state (h_t)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t    


class KinoDynamicsCoMRSSM(nn.Module):
    def __init__(
        self,
        task_cfg, 
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        embed=512,
        device="cpu",
    ):
        super(KinoDynamicsCoMRSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        # Task specific parameters
        self._task_cfg = task_cfg 
        self._actions_dim: int = task_cfg["num_actions"]
        self._joint_pos_dim: int = task_cfg["num_joint_pos"]
        self._joint_vel_dim: int = task_cfg["num_joint_vel"]
        self._joint_state_dim: int = self._joint_pos_dim + self._joint_vel_dim
        self._action_transform_dim: int = 6 + self._joint_vel_dim # force + torque + joint_vel_delta

        # Constants
        self._com_state_dim : int = 3 + 3 + 3 + 3 # p + rpy + lin_vel + ang_vel

        self.gravity = [0, 0, -9.81] # m/s^2

        self.dt = 1/5 # time step.

        self._com_mass_sqrt = nn.Parameter(torch.ones(1))
        self._com_intertia_sqrt = nn.Parameter(torch.ones(3))


        # GRU Step
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + self._com_state_dim + self._joint_state_dim + self._actions_dim 
        else:
            inp_dim = self._stoch + self._com_state_dim + self._joint_state_dim + self._actions_dim
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)


        # Prior
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)


        # Prior CoM Joint dynamics updated using action transform
        act_trans_out_layers = []
        inp_dim = self._deter
        act_trans_out_layers.append(nn.Linear(inp_dim, self._action_transform_dim, bias=False))
        if norm:
            act_trans_out_layers.append(nn.LayerNorm(self._action_transform_dim, eps=1e-03))
        act_trans_out_layers.append(act())
        self._act_trans_out_layers = nn.Sequential(*act_trans_out_layers)
        self._act_trans_out_layers.apply(tools.weight_init)


        # Post
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)


        # Post CoM Joint Estimation
        com_joint_out_layers = []
        inp_dim = self._deter + self._embed
        com_joint_out_layers.append(nn.Linear(inp_dim, self._com_state_dim + self._joint_state_dim, bias=False))
        if norm:
            com_joint_out_layers.append(nn.LayerNorm(self._com_state_dim + self._joint_state_dim, eps=1e-03))
        com_joint_out_layers.append(act())
        self._com_joint_out_layers = nn.Sequential(*com_joint_out_layers)
        self._com_joint_out_layers.apply(tools.weight_init)


        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size, embed):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        com_joint = self.get_com_joint(deter, embed)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
                com_joint=com_joint
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
                com_joint=com_joint,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            state["com_joint"] = self.get_com_joint(state["deter"], embed)
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"], state['com_joint']], -1)

    def get_deter_feat(self, state):
        return state["deter"]

    def get_com_joint_feat(self, state):
        return state["com_joint"]

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def check_reset(self, is_first, embed=None, prev_state=None, prev_action=None):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first), embed)
            if prev_action is not None:
                prev_action = torch.zeros((len(is_first), self._actions_dim)).to(
                    self._device
                )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            if prev_action is not None:
                prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first), embed)
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )
        
        return prev_state, prev_action

    def check_state_reset_with_masks(self, is_first, embed, prev_state=None):
        batch_size = len(is_first)
        init_state = self.initial(batch_size, embed)

        if prev_state is None: return init_state

        prev_state_out = prev_state
        for key, val in prev_state_out.items():
            is_first_r = torch.reshape(
                is_first,
                is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
            )
            prev_state_out[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r

        return prev_state_out

    def check_action_reset_with_masks(self, is_first, prev_action=None):
        batch_size = len(is_first)
        init_action = torch.zeros((batch_size, self._actions_dim)).to(self._device)
    
        if prev_action is None: return init_action

        is_first_exp = is_first[:, None]
        prev_action_out = prev_action * (1.0 - is_first_exp) + init_action * is_first_exp

        return prev_action_out

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        prev_state, prev_action = self.check_reset(is_first, embed, prev_state, prev_action)

        prior = self.img_step(prev_state, prev_action)

        post = self.post_step(prior["deter"], embed, sample=sample)

        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)

        x = torch.cat([prev_stoch, prev_state['com_joint'], prev_action], -1)

        # GRU
        # (batch, stoch * discrete_num + com_joint + action) -> (batch, hidden)
        x_gru = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            new_deter, deter = self._cell(x_gru, [deter])
            deter = deter[0]  # Keras wraps the state in a list.

        # q_theta
        # (batch, deter) -> (batch, hidden)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", self._img_out_layers(new_deter))
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}

        # CoM Joint dynamics
        # (batch, deter) -> (batch, action_transform_dim)
        action_transform = self._act_trans_out_layers(new_deter)
        # (batch, com_state_dim + joint_state_dim)
        prior["com_joint"] = self.eom(prev_state["com_joint"], action_transform)

        return prior

    def post_step(self, deter, embed, sample=True):
        # p_theta

        x = torch.cat([deter, embed], -1)
        # (batch, deter + embed) -> (batch, hidden)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", self._obs_out_layers(x))
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": deter, **stats}

        # Posterior CoM Joint estimation
        # (batch_size, prior_deter + embed) -> (batch_size, com_state_dim + joint_state_dim)
        post["com_joint"] = self._com_joint_out_layers(x)
        post["com_joint"][..., 0:2] = 0. # CoM position x,y is set to 0,0

        return post

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()
    
    def get_com_joint(self, deter, embed):
        return self._com_joint_out_layers(torch.cat([deter, embed], -1))

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def eom(self, com_joint_state, action_transform):
        # equations of motions.
        # TODO: wrap around joint limits & [-pi, pi].
        
        mass = torch.square(self._com_mass_sqrt) + 1e-6
        inertia = torch.square(self._com_intertia_sqrt) + 1e-6
        
        com_state = com_joint_state[..., :self._com_state_dim]
        joint_state = com_joint_state[..., self._com_state_dim:]

        p = com_state[..., :3]
        theta = com_state[..., 3:6]
        lin_vel = com_state[..., 6:9]
        ang_vel = com_state[..., 9:12]
        
        force = action_transform[..., :3]
        torque = action_transform[..., 3:6]
        joint_vel_delta = action_transform[..., 6 : 6 + self._joint_state_dim]
        
        joint_pos = joint_state[..., :self._joint_pos_dim]
        joint_vel = joint_state[..., self._joint_pos_dim: self._joint_pos_dim + self._joint_vel_dim]

        # CoM Dynamics       
        # continuous dynamics.
        ang_acc = (1/inertia) * (-tools.cross(ang_vel, inertia * ang_vel) + torque)
        lin_acc = (1/mass) * force + torch.tensor(self.gravity).to(com_state.device)
        
        # integrate dynamics.
        new_ang_vel = ang_vel + ang_acc * self.dt
        new_lin_vel = lin_vel + lin_acc * self.dt

        new_theta_dot = tools.angvel_to_rpydot(theta, ang_vel)
        new_theta = theta + new_theta_dot * self.dt
        new_p = p + lin_vel * self.dt

        new_com_state = torch.cat([new_p, new_theta, new_lin_vel, new_ang_vel], dim=-1)

        # Kinematics: TODO: handle wheels
        new_joint_vel = joint_vel + joint_vel_delta
        if self._task_cfg['wheel'] is False:
            new_joint_pos = joint_pos + new_joint_vel * self.dt
        else:
            # Assume index [3,7] are wheels
            joint_pos_indices = [0, 1, 2, 4, 5, 6]
            new_joint_pos = joint_pos + new_joint_vel[..., joint_pos_indices] * self.dt

        new_joint_state = torch.cat([new_joint_pos, new_joint_vel], dim=-1)

        return torch.cat([new_com_state, new_joint_state], dim=-1)

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss

    def com_joint_loss(self, post, prior, data):
        # data['com_state']: (batch, time, com_state_dim)
        # data['joint_state']: (batch, time, joint_state_dim)

        com_position_ref = data["com_state"][:, 0:1, :2].clone()
        # because we don't have global positioning system on hw, predict com with respect to the first frame
        com_prior_target = data["com_state"].clone()
        com_prior_target[..., :2] -= com_position_ref

        if self._task_cfg['wheel'] is False:
            joint_prior_target = data["joint_state"].clone()
        else:
            # Assume index [3,7] are wheels
            joint_pos_indices = [0, 1, 2, 4, 5, 6]
        
            joint_prior_target = torch.concatenate(
                [data["joint_state"][..., joint_pos_indices],
                data["joint_state"][..., self._joint_pos_dim + 2: self._joint_pos_dim + 2 + self._joint_vel_dim] # 2 for the wheels
                ], dim=-1
            )


        # Prior CoM Joint dynamics
        com_pred = prior["com_joint"][..., :self._com_state_dim]
        joint_pred = prior["com_joint"][..., self._com_state_dim:]
        loss_prior = F.mse_loss(com_pred, com_prior_target) + F.mse_loss(joint_pred, joint_prior_target)


        com_post_target = data["com_state"].clone()
        com_post_target[..., :2] = 0.
        if self._task_cfg['wheel'] is False:
            joint_post_target = data["joint_state"].clone()
        else:
            # Assume index [3,7] are wheels
            joint_pos_indices = [0, 1, 2, 4, 5, 6]
        
            joint_post_target = torch.concatenate(
                [data["joint_state"][..., joint_pos_indices],
                data["joint_state"][..., self._joint_pos_dim + 2: self._joint_pos_dim + 2 + self._joint_vel_dim] # 2 for the wheels
                ], dim=-1
            )

        # Post CoM Joint estimation
        com_post = post["com_joint"][..., :self._com_state_dim]
        joint_post = post["com_joint"][..., self._com_state_dim:]
        loss_post = F.mse_loss(com_post, com_post_target) + F.mse_loss(joint_post, joint_post_target)


        total_loss = loss_prior + loss_post

        return total_loss, loss_prior.mean(), loss_post.mean()