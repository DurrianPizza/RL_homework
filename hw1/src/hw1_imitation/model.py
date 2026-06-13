"""Model definitions for Push-T imitation policies."""

from __future__ import annotations
import math
import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        upper_dim = self.state_dim * self.state_dim
        self.projection = nn.Linear(state_dim, upper_dim * chunk_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=upper_dim, nhead=1, dim_feedforward=hidden_dims[0],
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.output_layer = nn.Sequential(
            *[nn.Linear(upper_dim, state_dim), nn.ReLU(), nn.Linear(self.state_dim, action_dim)])

    def forward(self,
                state: torch.Tensor,
                *_):
        projection = self.projection(state)
        projection = projection.view(-1, self.chunk_size, self.state_dim * self.state_dim)
        x = self.decoder(projection, projection.clone())
        # reshape 为动作块
        action_chunk = self.output_layer(x)
        return action_chunk

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) — one scalar time per sample
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss.

    Architecture: per-chunk-position MLP that takes
        [state_enc(state), time_enc(t), action_t_chunk]
    and outputs the velocity field v(a_t, t | state).
    dt is sampled as one scalar per sample and broadcast across
    chunk positions and action dims.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dim = hidden_dims[0]

        # state encoder: state -> per-token context
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # time embedding: scalar t -> sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # velocity net: chunk-internal self-attention + per-position MLP head
        # 用 TransformerDecoder(self, self.clone()) 等价于 encoder + 一组独立的 cross-attn,
        # 跟 MSEPolicy 的 decoder(projection, projection.clone()) 架构对齐
        self.token_dim = 2 * self.hidden_dim + action_dim
        self.attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.token_dim,
                nhead=1,
                dim_feedforward=hidden_dims[0],
                batch_first=True,
            ),
            num_layers=10,
        )
        self.head = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], action_dim),
        )

    def generate_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, self.chunk_size, self.action_dim))

    def generate_tao(self, batch_size: int) -> torch.Tensor:
        # one scalar t per sample, broadcast across chunk and action_dim
        return torch.rand((batch_size, 1, 1))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        # state: (B, state_dim); action: (B, T, A); dt: (B, 1, 1)
        state_feat = self.state_encoder(state).unsqueeze(1)        # (B, 1, H)
        time_feat = self.time_embed(dt.reshape(-1)).unsqueeze(1)   # (B, 1, H)
        state_feat = state_feat.expand(-1, self.chunk_size, -1)    # (B, T, H)
        time_feat = time_feat.expand(-1, self.chunk_size, -1)     # (B, T, H)
        tokens = torch.cat([state_feat, time_feat, action], dim=-1)  # (B, T, 2H + A)
        tokens = self.attn(tokens, tokens.clone())                  # (B, T, 2H + A)
        return self.head(tokens)                                    # (B, T, A)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        device = action_chunk.device
        a_0 = self.generate_noise(action_chunk.shape[0]).to(device)  # (B, T, A)
        dt = self.generate_tao(action_chunk.shape[0]).to(device)     # (B, 1, 1)
        a_t = dt * action_chunk + (1.0 - dt) * a_0                   # (B, T, A)
        velocity = self.forward(state, a_t, dt)
        target = action_chunk - a_0
        return nn.functional.mse_loss(velocity, target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        device = state.device
        noise = self.generate_noise(state.shape[0]).to(device)    # (B, T, A)
        dt_step = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full(
                (state.shape[0], 1, 1),
                step * dt_step,
                device=device,
                dtype=state.dtype,
            )
            velocity = self.forward(state, noise, t)
            noise = noise + velocity * dt_step
        return noise



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


if __name__ == "__main__":
    model = FlowMatchingPolicy(state_dim=8, action_dim=2, chunk_size=8, hidden_dims=(8, 8))
    batch_size = 128
    input_state = torch.rand((batch_size, 8))
    input_action_chunk = torch.rand((batch_size, 8, 2))

    model.compute_loss(input_state, input_action_chunk)
    model.sample_actions(input_state, num_steps=10)
    print(model)
