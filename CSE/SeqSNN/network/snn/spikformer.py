from typing import Optional

from pathlib import Path
import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from ..base import NETWORKS
from ...module.positional_encoding import PositionEmbedding
from ...module.spike_encoding import SpikeEncoder
from ...module.spike_attention import Block

# clustering
from ...module.clustering import Cluster_assigner, cluster_base_encoding

tau = 2.0  # beta = 1 - 1/tau
backend = "torch"
detach_reset = True
threshold = 0.085


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, D, L
        enc = self.encoder(inputs)  # B, T, D, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, D, L
        spks = self.lif(enc)  # T, B, D, L
        return spks


@NETWORKS.register_module("Spikformer")
class Spikformer(nn.Module):
    _snn_backend = "spikingjelly"

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        num_pe_neuron: int = 10,
        pe_type: str = "none",
        pe_mode: str = "concat",  # "add" or concat
        neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
        depths: int = 2,
        common_thr: float = 0.775,
        max_length: int = 5000,
        num_steps: int = 4,
        heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        input_size: Optional[int] = None,
        weight_file: Optional[Path] = None,
        # clustering
        use_cluster: bool = False,
        n_cluster: int = 3,
        d_model: int = 128,
        cluster_method: str = 'attention',
        device: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron

        self.temporal_encoder = SpikeEncoder[self._snn_backend]["conv"](num_steps)
        self.pe = PositionEmbedding(
            pe_type=pe_type,
            pe_mode=pe_mode,
            neuron_pe_scale=neuron_pe_scale,
            input_size=input_size,
            max_len=max_length,
            num_pe_neuron=self.num_pe_neuron,
            dropout=0.1,
            num_steps=num_steps,
        )
        if (self.pe_type == "neuron" and self.pe_mode == "concat") or (
            self.pe_type == "random" and self.pe_mode == "concat"
        ):
            self.encoder = nn.Linear(input_size + num_pe_neuron, dim)
        else:
            self.encoder = nn.Linear(input_size, dim)
        self.init_lif = neuron.LIFNode(
            tau=tau,
            step_mode="m",
            detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr,
            backend=backend,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    length=max_length,
                    tau=tau,
                    common_thr=common_thr,
                    dim=dim,
                    d_ff=self.d_ff,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                )
                for _ in range(depths)
            ]
        )

        self.use_cluster = use_cluster
        self.device = device
        if self.use_cluster:
            self.cluster_assigner = Cluster_assigner(
                n_vars=input_size,
                n_cluster=n_cluster,
                seq_len=max_length,
                d_model=d_model,
                device=self.device,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, inputs):
        functional.reset_net(self)

        spike_counter = []
        hidden_shape = []

        hiddens = self.temporal_encoder(inputs)  # B L C -> T B C L

        if self.use_cluster:
            cluster_prob, cluster_emb, x_emb = self.cluster_assigner(
                inputs, self.cluster_assigner.cluster_emb
            )
            cluster_prob_hard = cluster_base_encoding(cluster_prob, hiddens.size(3))

            hiddens = torch.cat((hiddens, cluster_prob_hard), dim=0)

        hiddens = hiddens.transpose(-2, -1)  # T B L C
        if self.pe_type != "none":
            hiddens = self.pe(hiddens)  # T B L C'
        T, B, L, _ = hiddens.shape

        hiddens = self.encoder(hiddens.flatten(0, 1)).reshape(T, B, L, -1)  # T B L D
        hiddens = self.init_lif(hiddens)

        for blk in self.blocks:
            hiddens = blk(hiddens)  # T B L D

        out = hiddens.mean(0)
        total_spike = sum(spike_counter)
        return out, out.mean(dim=1)  # B L D, B D

    @property
    def output_size(self):
        return self.dim

    @property
    def hidden_size(self):
        return self.dim
