import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from ..layers import Attention, RMSNorm, RotaryEmbedding, SiluAndMul
from ..layers.attention import AttnInputs
from ..layers.qk_norm_rope import qk_norm_rope
from .loader import default_weight_loader


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
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
        self.num_heads, self.num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
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
        q, k = qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            positions,
            rotary_emb.cos_sin_cache,
            self.q_norm.eps,
        )
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
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding.from_config(config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttnInputs):
        h, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, residual = layer(positions, h, residual, attn, self.rotary_emb)
        return self.norm(h, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        tied = getattr(config, "tie_word_embeddings", False)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device="meta" if tied else None
        )
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, positions, attn: AttnInputs):
        return self.model(input_ids, positions, attn)

    def compute_logits(self, hidden_states, out=None):
        if out is not None:
            return torch.mm(hidden_states, self.lm_head.weight.t(), out=out)
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        skipped = []

        attn = self.model.layers[0].self_attn
        q, kv = attn.q_size, attn.kv_size
        intermediate = self.model.layers[0].mlp.down_proj.in_features
        slices = {
            "q_proj": ("qkv_proj", 0, q),
            "k_proj": ("qkv_proj", q, kv),
            "v_proj": ("qkv_proj", q + kv, kv),
            "gate_proj": ("gate_up_proj", 0, intermediate),
            "up_proj": ("gate_up_proj", intermediate, intermediate),
        }
        for name, weight in weights:
            projection = name.split(".")[-2]
            if projection in slices:
                target, offset, size = slices[projection]
                target_name = name.replace(projection, target)
                params[target_name].data[offset : offset + size].copy_(weight)
                loaded.add(target_name)
            elif name in params:
                default_weight_loader(params[name], weight)
                loaded.add(name)
            else:
                skipped.append(name)

        unloaded = set(params) - loaded
        if "lm_head.weight" in unloaded and "model.embed_tokens.weight" in loaded:
            default_weight_loader(params["lm_head.weight"], params["model.embed_tokens.weight"])
            loaded.add("lm_head.weight")
            unloaded.discard("lm_head.weight")

        if unloaded:
            print(f"[WARN] {len(unloaded)} params NOT loaded: {sorted(unloaded)[:10]}...")
        if skipped:
            print(f"[INFO] {len(skipped)} HF weights skipped (not in model): {skipped[:5]}...")
