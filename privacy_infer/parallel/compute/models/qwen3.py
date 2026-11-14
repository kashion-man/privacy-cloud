import sys

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..layers import (
    Attention,
    MergedColumnParallelLinear,
    ParallelLMHead,
    QKVParallelLinear,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    SiluAndMul,
    VocabParallelEmbedding,
)
from ..layers.attention import AttnInputs
from ..layers.linear import get_weight_loader
from ...distributed import IntermediateTensors, PPMissingLayer, get_pp_group, get_pp_indices


def _parse_layer_idx(name: str) -> int | None:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    idx = rest.split(".", 1)[0]
    try:
        return int(idx)
    except ValueError:
        return None


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False
        )

        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, input_is_parallel=True
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=getattr(config, "attention_bias", False),
        )
        self.num_heads, self.num_kv_heads = self.qkv_proj.num_heads, self.qkv_proj.num_kv_heads
        self.o_proj = RowParallelLinear(
            self.qkv_proj.num_heads_total * self.head_dim,
            config.hidden_size,
            input_is_parallel=True,
        )
        self.q_size, self.kv_size = (
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads, layer_id=layer_id)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, rotary_emb, attn):
        q, k, v = self.qkv_proj(hidden_states).split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        q = self.q_norm(q)[0]
        k = self.k_norm(k)[0]
        q, k = rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v, attn))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_id: int = 0):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_id)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions, hidden_states, residual: torch.Tensor | None, attn: AttnInputs, rotary_emb
    ):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb, attn)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        pp = get_pp_group()
        self.rotary_emb = RotaryEmbedding.from_config(config)

        self.start_layer, self.end_layer = get_pp_indices(
            config.num_hidden_layers, pp.group_rank, pp.size
        )

        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if pp.is_first_rank
            else PPMissingLayer()
        )

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_id=local_i)
                for local_i in range(self.end_layer - self.start_layer)
            ]
        )

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if pp.is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttnInputs,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        if intermediate_tensors is None:
            h, residual = self.embed_tokens(input_ids), None
        else:
            h = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers:
            h, residual = layer(positions, h, residual, attn, self.rotary_emb)

        if get_pp_group().is_last_rank:
            return self.norm(h, residual)[0]

        return IntermediateTensors(hidden_states=h, residual=residual)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings

        self.lm_head = (
            ParallelLMHead(config.hidden_size, config.vocab_size, bias=False)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttnInputs,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, attn, intermediate_tensors)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)

    def argmax(self, hidden_states):
        return self.lm_head.argmax(hidden_states)

    def load_weights(self, weights):
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded = set()
        skipped = []
        tied_embeddings = None

        for name, w in weights:
            if name == "model.embed_tokens.weight" and self.tie_word_embeddings:
                tied_embeddings = w
            layer_idx = _parse_layer_idx(name)
            if layer_idx is not None:
                if not (self.model.start_layer <= layer_idx < self.model.end_layer):
                    continue

                name = name.replace(
                    f"model.layers.{layer_idx}.",
                    f"model.layers.{layer_idx - self.model.start_layer}.",
                    1,
                )

            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name:
                    continue

                our = name.replace(weight_name, param_name)
                if our in params:
                    params[our].weight_loader(params[our], w, shard_id)
                    loaded.add(our)
                else:
                    skipped.append(name)
                break

            else:
                if name in params:
                    get_weight_loader(params[name])(params[name], w)
                    loaded.add(name)
                else:
                    skipped.append(name)

        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and tied_embeddings is not None:
            self.lm_head.weight_loader(params["lm_head.weight"], tied_embeddings)
            unloaded.discard("lm_head.weight")

        if unloaded:
            raise ValueError(f"Missing model parameters: {sorted(unloaded)[:10]}")
        if skipped:
            print(f"[INFO] {len(skipped)} HF weights skipped (not in model): {skipped[:5]}...",
                  file=sys.stderr)
