# 1_manual_forward_pass.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench import measure_perplexity

print("Loading tokenizer and reference model...")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"

# We load in fp16 because run_baseline.py generated ground_logits.pt in fp16
hf_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
sd = hf_model.state_dict()

# --- Qwen2.5-0.5B Config ---
hidden_size = 896
intermediate_size = 4864
num_layers = 24
num_heads = 14
num_kv_heads = 2
head_dim = 64
rope_theta = 1000000.0
rms_norm_eps = 1e-6

# --- Helper Functions ---
def rmsnorm(x, weight):
    # HF computes RMSNorm in fp32 to prevent fp16 overflow (x^2 can exceed 65504)
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    return (weight * x).to(input_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_iope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ManualQwen2(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd
        self.cos = None
        self.sin = None

    def precompute_rope(self, seq_len, device):
        dim = head_dim
        freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Keep cos/sin in fp32 for precision during RoPE application
        self.cos = emb.cos().view(1, 1, seq_len, head_dim)
        self.sin = emb.sin().view(1, 1, seq_len, head_dim)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if self.cos is None or self.cos.shape[2] < seq_len:
            self.precompute_rope(seq_len, device)
            
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]

        hidden_states = F.embedding(input_ids, self.sd["model.embed_tokens.weight"])
        
        # HF uses torch.finfo.min instead of -inf for causal masks
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, device=device, dtype=torch.float32), 
            diagonal=1
        )

        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            normed = rmsnorm(hidden_states, self.sd[prefix + "input_layernorm.weight"])
            
            q_proj = F.linear(normed, self.sd[prefix + "self_attn.q_proj.weight"])
            k_proj = F.linear(normed, self.sd[prefix + "self_attn.k_proj.weight"])
            v_proj = F.linear(normed, self.sd[prefix + "self_attn.v_proj.weight"])

            q = q_proj.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Cast to fp32 for RoPE to prevent precision loss
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            
            q, k = apply_iope(q, k, cos, sin)
            
            # GQA broadcasting (14 heads / 2 kv_heads = 7 repeats)
            k = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            
            # Attention matmul in fp32 to prevent overflow in dot product
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            
            attn_output = torch.matmul(attn_weights, v.to(torch.float32))
            attn_output = attn_output.to(v.dtype) # Back to fp16
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            attn_output = F.linear(attn_output, self.sd[prefix + "self_attn.o_proj.weight"])
            
            hidden_states = hidden_states + attn_output
            
            normed = rmsnorm(hidden_states, self.sd[prefix + "post_attention_layernorm.weight"])
            
            gate = F.linear(normed, self.sd[prefix + "mlp.gate_proj.weight"])
            up = F.linear(normed, self.sd[prefix + "mlp.up_proj.weight"])
            mlp_out = F.linear(F.silu(gate) * up, self.sd[prefix + "mlp.down_proj.weight"])
            
            hidden_states = hidden_states + mlp_out

        hidden_states = rmsnorm(hidden_states, self.sd["model.norm.weight"])
        
        # tie_word_embeddings is True for Qwen2.5-0.5B
        lm_head_w = self.sd["model.embed_tokens.weight"]
        logits = F.linear(hidden_states, lm_head_w)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        class DummyOut: pass
        out = DummyOut()
        out.logits = logits
        out.loss = loss
        return out

print("Building manual model and loading weights...")
manual_model = ManualQwen2(sd)
manual_model.eval()
manual_model.to(device)

# --- 1. Logits Comparison (Using exact prompt from run_baseline.py) ---
msgs = [{"role": "user", "content": "are u stupid?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
ids = ids.to(device)

print("Running manual forward...")
with torch.no_grad():
    manual_out = manual_model(ids)
    your_logits = manual_out.logits

ground = torch.load("ground_logits.pt", map_location=device)
max_diff = (your_logits - ground).abs().max().item()

print("\n--- LOGITS VERIFICATION ---")
print(f"Max absolute difference: {max_diff}")
if max_diff < 1e-3:
    print("✅ PASS! Your manual forward pass matches HuggingFace exactly.")
else:
    print("❌ MISMATCH! Check RoPE, GQA broadcasting, or RMSNorm.")

# --- 2. Perplexity Comparison (Using exact text from run_baseline.py) ---
print("\n--- PERPLEXITY BENCHMARK ---")
FIXED_TEXT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."
fixed_ids = tok(FIXED_TEXT, return_tensors="pt").input_ids.to(device)

ppl_manual = measure_perplexity(manual_model, fixed_ids)
print(f"perplexity on fixed text: {ppl_manual:.3f}")
