# 1_manual_forward_pass.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench import measure_perplexity, measure_generate, print_report


print("Loading tokenizer and reference model...")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load HF model only to extract the fp16 state dict.
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
).to(device)

sd = hf_model.state_dict()

# Free the HF module memory; sd keeps references to the tensors.
del hf_model
if device == "cuda":
    torch.cuda.empty_cache()


# -----------------------------
# Qwen2.5-0.5B config
# -----------------------------
hidden_size = 896
intermediate_size = 4864
num_layers = 24
num_heads = 14
num_kv_heads = 2
head_dim = 64
rope_theta = 1000000.0
rms_norm_eps = 1e-6

# MLA latent bottleneck.
# 64 is chosen to make the KV cache visibly smaller than GQA.
mla_latent_dim = 64


# -----------------------------
# Helper functions
# -----------------------------
def rmsnorm(x, weight):
    # HF computes RMSNorm in fp32 to avoid fp16 overflow in x.pow(2).
    input_dtype = x.dtype
    x = x.to(torch.float32)

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)

    return (weight.float() * x).to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleCache:
    """
    Minimal cache object compatible with bench.py's KV-cache memory counting.

    For this MLA implementation:
      - key_cache stores the compressed latent c
      - value_cache stores the decoupled RoPE key k_rope
    """
    def __init__(self, key_cache=None, value_cache=None):
        self.key_cache = key_cache if key_cache is not None else []
        self.value_cache = value_cache if value_cache is not None else []


class ManualQwen2MLA(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd
        self.cos = None
        self.sin = None

    def precompute_rope(self, seq_len, device):
        dim = head_dim

        # RoPE frequencies.
        freqs = 1.0 / (
            rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos = emb.cos().view(1, 1, seq_len, head_dim)
        self.sin = emb.sin().view(1, 1, seq_len, head_dim)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # -----------------------------
        # KV cache handling
        # -----------------------------
        if past_key_values is None:
            past_length = 0
            past_layers = None
        else:
            # This program returns SimpleCache.
            # key_cache[i] shape: [B, past_len, mla_latent_dim]
            past_length = past_key_values.key_cache[0].shape[1]
            past_layers = list(
                zip(past_key_values.key_cache, past_key_values.value_cache)
            )

        total_seq_len = past_length + seq_len

        if self.cos is None or self.cos.shape[2] < total_seq_len:
            self.precompute_rope(total_seq_len, device)

        # RoPE only for the current query positions.
        cos = self.cos[:, :, past_length:total_seq_len, :]
        sin = self.sin[:, :, past_length:total_seq_len, :]

        hidden_states = F.embedding(
            input_ids,
            self.sd["model.embed_tokens.weight"],
        )

        # Causal mask for current query rows against all keys.
        causal_mask = torch.triu(
            torch.full(
                (total_seq_len, total_seq_len),
                torch.finfo(torch.float32).min,
                device=device,
                dtype=torch.float32,
            ),
            diagonal=1,
        )[past_length:total_seq_len, :]

        new_key_cache = [] if use_cache else None
        new_value_cache = [] if use_cache else None

        # -----------------------------
        # Transformer layers
        # -----------------------------
        for i in range(num_layers):
            prefix = f"model.layers.{i}."

            # Pre-attention RMSNorm.
            normed = rmsnorm(
                hidden_states,
                self.sd[prefix + "input_layernorm.weight"],
            )

            # ==================================================
            # MULTI-HEAD LATENT ATTENTION CORE
            # ==================================================
            #
            # True MLA has dedicated low-rank weights.
            # Qwen2.5 does not, so existing Qwen weights are reused
            # as stand-ins to demonstrate the MLA computation pattern.
            #
            # MLA idea:
            #   1. Compress hidden_states into latent c.
            #   2. Derive Q/K/V from c.
            #   3. Use decoupled RoPE on a small separate key part.
            #   4. Cache only c and k_rope, not full expanded K/V.
            # ==================================================

            # 1. Latent compression / down-projection.
            # Reuse first mla_latent_dim rows of k_proj as W_down.
            W_down = self.sd[prefix + "self_attn.k_proj.weight"][
                :mla_latent_dim,
                :,
            ]
            c = F.linear(normed, W_down)
            c = c + self.sd[prefix + "self_attn.k_proj.bias"][:mla_latent_dim]

            # 2. Query nope part from latent c.
            # Q_nope does not receive RoPE.
            W_uq = self.sd[prefix + "self_attn.q_proj.weight"][
                :,
                :mla_latent_dim,
            ]
            q_nope = F.linear(c, W_uq)

            # 3. Decoupled RoPE parts.
            # These are computed from the normalized hidden state directly.
            W_qr = self.sd[prefix + "self_attn.q_proj.weight"][:head_dim, :]
            W_kr = self.sd[prefix + "self_attn.k_proj.weight"][:head_dim, :]

            q_rope = F.linear(normed, W_qr)
            q_rope = q_rope + self.sd[prefix + "self_attn.q_proj.bias"][:head_dim]

            k_rope = F.linear(normed, W_kr)
            k_rope = k_rope + self.sd[prefix + "self_attn.k_proj.bias"][:head_dim]

            # Reshape current-token Q and RoPE parts.
            q_nope = q_nope.view(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2)

            q_rope = q_rope.view(
                batch_size,
                seq_len,
                1,
                head_dim,
            ).transpose(1, 2)

            k_rope = k_rope.view(
                batch_size,
                seq_len,
                1,
                head_dim,
            ).transpose(1, 2)

            # RoPE math is safer in fp32.
            q_nope = q_nope.to(torch.float32)
            q_rope = q_rope.to(torch.float32)
            k_rope = k_rope.to(torch.float32)

            # Apply RoPE only to the decoupled rope parts.
            q_rope, k_rope = apply_rope(q_rope, k_rope, cos, sin)

            # 4. Compressed KV cache.
            # Cache:
            #   - c: latent bottleneck
            #   - k_rope: RoPE-applied decoupled key
            past_kv = None if past_layers is None else past_layers[i]

            if past_kv is not None:
                past_c, past_k_rope = past_kv

                c_full = torch.cat(
                    [past_c.to(c.dtype), c],
                    dim=1,
                )

                k_rope_full = torch.cat(
                    [past_k_rope.to(torch.float32), k_rope],
                    dim=2,
                )
            else:
                c_full = c
                k_rope_full = k_rope

            if use_cache:
                # bench.py expects .key_cache and .value_cache lists.
                # Here:
                #   key_cache   = compressed latent c
                #   value_cache = decoupled RoPE key k_rope
                new_key_cache.append(c_full.to(torch.float16))
                new_value_cache.append(k_rope_full.to(torch.float16))

            # 5. Reconstruct full K_nope and V from cached latent c.
            # This is the tradeoff: smaller cache, more recompute.
            W_uk = self.sd[prefix + "self_attn.q_proj.weight"][
                :,
                mla_latent_dim : 2 * mla_latent_dim,
            ]

            W_uv = self.sd[prefix + "self_attn.v_proj.weight"][
                :mla_latent_dim,
                :,
            ].T

            k_nope_full = F.linear(c_full, W_uk)
            v_full = F.linear(c_full, W_uv)

            k_nope_full = k_nope_full.view(
                batch_size,
                total_seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2).to(torch.float32)

            v_full = v_full.view(
                batch_size,
                total_seq_len,
                num_heads,
                head_dim,
            ).transpose(1, 2).to(torch.float32)

            # 6. MLA attention score:
            # score = Q_nope @ K_nope^T + Q_rope @ K_rope^T
            attn_weights_nope = torch.matmul(
                q_nope,
                k_nope_full.transpose(2, 3),
            )

            attn_weights_rope = torch.matmul(
                q_rope,
                k_rope_full.transpose(2, 3),
            )

            attn_weights = (
                attn_weights_nope + attn_weights_rope
            ) / math.sqrt(head_dim)

            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(
                attn_weights,
                dim=-1,
                dtype=torch.float32,
            )

            attn_output = torch.matmul(attn_weights, v_full)
            attn_output = attn_output.to(torch.float16)

            # ==================================================
            # END MLA CORE
            # ==================================================

            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size,
                seq_len,
                hidden_size,
            )

            attn_output = F.linear(
                attn_output,
                self.sd[prefix + "self_attn.o_proj.weight"],
            )

            # Attention residual.
            hidden_states = hidden_states + attn_output

            # MLP block.
            normed = rmsnorm(
                hidden_states,
                self.sd[prefix + "post_attention_layernorm.weight"],
            )

            gate = F.linear(
                normed,
                self.sd[prefix + "mlp.gate_proj.weight"],
            )
            up = F.linear(
                normed,
                self.sd[prefix + "mlp.up_proj.weight"],
            )

            mlp_out = F.linear(
                F.silu(gate) * up,
                self.sd[prefix + "mlp.down_proj.weight"],
            )

            # MLP residual.
            hidden_states = hidden_states + mlp_out

        # Final norm.
        hidden_states = rmsnorm(
            hidden_states,
            self.sd["model.norm.weight"],
        )

        # Qwen2.5-0.5B ties word embeddings.
        lm_head_w = self.sd["model.embed_tokens.weight"]
        logits = F.linear(hidden_states, lm_head_w)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        class DummyOut:
            pass

        out = DummyOut()
        out.logits = logits
        out.loss = loss

        if use_cache:
            out.past_key_values = SimpleCache(
                key_cache=new_key_cache,
                value_cache=new_value_cache,
            )
        else:
            out.past_key_values = None

        return out


print("Building manual MLA model and loading weights...")

manual_model = ManualQwen2MLA(sd)
manual_model.eval()
manual_model.to(device)


# -----------------------------
# 1. Logits verification
# -----------------------------
# This must match the prompt used by run_baseline.py to create ground_logits.pt.
VERIFY_PROMPT = "write random words"

msgs_verify = [
    {
        "role": "user",
        "content": VERIFY_PROMPT,
    }
]

ids_verify = tok.apply_chat_template(
    msgs_verify,
    add_generation_prompt=True,
    return_tensors="pt",
)

if hasattr(ids_verify, "input_ids"):
    ids_verify = ids_verify.input_ids

ids_verify = ids_verify.to(device)

print("Running manual forward for logits verification...")

with torch.no_grad():
    manual_out = manual_model(ids_verify)
    your_logits = manual_out.logits

ground = torch.load("ground_logits.pt", map_location=device)

print("\n--- LOGITS VERIFICATION ---")
print(f"manual logits shape: {tuple(your_logits.shape)}")
print(f"ground logits shape: {tuple(ground.shape)}")

if your_logits.shape != ground.shape:
    print("❌ SHAPE MISMATCH!")
    print("run_baseline.py must generate ground_logits.pt using the exact same prompt:")
    print(f'"{VERIFY_PROMPT}"')
else:
    max_diff = (your_logits - ground).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    if max_diff < 1e-3:
        print("✅ PASS! Manual logits match ground logits.")
    else:
        print("❌ MISMATCH!")
        print("Note: MLA is not Qwen2.5's native attention, so a large diff is expected")
        print("unless the MLA weights are actually trained/converted for this model.")


# -----------------------------
# 2. Generation benchmark
# -----------------------------
print("\n--- GENERATION BENCHMARK ---")

attn_mask_verify = torch.ones_like(ids_verify)

metrics_manual = measure_generate(
    manual_model,
    tok,
    ids_verify,
    attn_mask_verify,
    max_new_tokens=50,
)

print(tok.decode(metrics_manual["output_ids"][0], skip_special_tokens=True))
print_report("manual forward (MLA)", metrics_manual)


# -----------------------------
# 3. Perplexity benchmark
# -----------------------------
print("\n--- PERPLEXITY BENCHMARK ---")

FIXED_TEXT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."

fixed_ids = tok(FIXED_TEXT, return_tensors="pt").input_ids.to(device)

ppl_manual = measure_perplexity(manual_model, fixed_ids)

print(f"perplexity on fixed text: {ppl_manual:.3f}")
