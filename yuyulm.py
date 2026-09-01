"""Tiny chat LM: raw PyTorch training/inference, designed for a 4 GB RTX 2050."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# [BOILERPLATE] HF's fast tokenizer supplies mature Rust BPE logic; writing BPE here
# would add a large amount of unrelated code and would not improve GPU inference.
from transformers import AutoTokenizer
# [BOILERPLATE] The dataset loader handles streaming, caching, and Hub formats.
from datasets import load_dataset

MAX_SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    vocab: int = 32000
    dim: int = 512
    layers: int = 10
    heads: int = 8
    ff: int = 1408
    max_seq_len: int = 1024
    dropout: float = 0.0


def rmsnorm(x, weight, eps=1e-6):
    # [INFERENCE FOCUS] RMSNorm avoids LayerNorm's mean/variance bookkeeping and
    # uses one reduction plus elementwise operations, reducing memory traffic.
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * weight


def rope(x, start):
    # [INFERENCE FOCUS] Complex numbers express the paired 2-D rotations compactly;
    # the angles are generated on the fly and cached nowhere, keeping the model small.
    b, h, t, d = x.shape
    half = d // 2
    pos = torch.arange(start, start + t, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    z = torch.view_as_complex(x.float().reshape(b, h, t, half, 2))
    rot = torch.polar(torch.ones_like(pos[:, None]), pos[:, None] * inv[None, :])
    return torch.view_as_real(z * rot[None, None]).flatten(-2).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.d = c.heads, c.dim // c.heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x, past=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(b, t, self.h, self.d).transpose(1, 2)
        k = k.view(b, t, self.h, self.d).transpose(1, 2)
        v = v.view(b, t, self.h, self.d).transpose(1, 2)
        old = 0 if past is None else past[0].size(2)
        q, k = rope(q, old), rope(k, old)
        # [INFERENCE FOCUS] This explicit concat is a memory copy. Production
        # engines use paged/ring KV storage; it is visible here for learning.
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # [INFERENCE FOCUS] Truncating bounds KV memory and attention work on 4 GB.
        if k.size(2) > MAX_SEQ_LEN:
            k, v = k[:, :, -MAX_SEQ_LEN:], v[:, :, -MAX_SEQ_LEN:]
            old = k.size(2) - t
        if old:
            allow = torch.ones(t, k.size(2), device=x.device, dtype=torch.bool)
            allow = torch.tril(allow, diagonal=old)
            allow = allow[None, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, -1)), (k, v)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.Parameter(torch.ones(c.dim))
        self.n2 = nn.Parameter(torch.ones(c.dim))
        self.attn, self.w1, self.w2, self.w3 = Attention(c), nn.Linear(c.dim, c.ff, False), nn.Linear(c.ff, c.dim, False), nn.Linear(c.dim, c.ff, False)

    def forward(self, x, past=None):
        a, cache = self.attn(rmsnorm(x, self.n1), past)
        x = x + a
        # [INFERENCE FOCUS] SwiGLU is silu(xW_gate) * (xW_up), with no GELU module.
        z = rmsnorm(x, self.n2)
        x = x + self.w2(F.silu(self.w1(z)) * self.w3(z))
        return x, cache


class YuYuLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.Parameter(torch.ones(c.dim))
        self.lm = nn.Linear(c.dim, c.vocab, False)
        self.lm.weight = self.tok.weight

    def forward(self, ids, past=None, targets=None):
        x = self.tok(ids)
        new = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if past is None else past[i])
            new.append(cache)
        logits = self.lm(rmsnorm(x, self.norm))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, tuple(new), loss

    @torch.no_grad()
    def generate(self, ids, max_new=80, temperature=.8, top_k=40):
        self.eval(); past = None
        for _ in range(max_new):
            # [INFERENCE FOCUS] First pass consumes the prompt; later passes consume
            # one token and reuse every prior K/V, changing O(T²) prompt work to O(T).
            logits, past, _ = self(ids if past is None else ids[:, -1:], past)
            p = logits[:, -1] / max(temperature, 1e-5)
            v, ix = torch.topk(p, min(top_k, p.size(-1)))
            probs = F.softmax(v, -1)
            ids = torch.cat((ids, ix.gather(-1, torch.multinomial(probs, 1))), 1)
        return ids


def train_one_epoch(model, tokenizer, steps=300, batch=2, seq=512, lr=3e-4):
    # [BOILERPLATE] SmolTalk provides conversational text; a small bounded slice
    # keeps this smoke-training run feasible on a 4 GB card.
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=.1)
    it = iter(ds)
    model.train()
    for step in range(steps):
        texts = []
        for _ in range(batch):
            row = next(it)
            msgs = row.get("messages", [])
            texts.append("\n".join(str(m.get("content", "")) for m in msgs))
        ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        _, _, loss = model(ids[:, :-1], targets=ids[:, 1:])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 25 == 0: print(f"step {step}: loss {loss.item():.3f}")


if __name__ == "__main__":
    # [AUTOMATION] Later wrap model construction with torch.compile() to fuse
    # RMSNorm/SwiGLU kernels and reduce Python dispatch overhead.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    cfg = Cfg(vocab=len(tokenizer), max_seq_len=MAX_SEQ_LEN)
    model = YuYuLM(cfg).to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M on {DEVICE}")
    # [AUTOMATION] Replace steps with a checkpointed scheduler and save every N
    # steps for unattended overnight training; this dummy run intentionally stays small.
    train_one_epoch(model, tokenizer, steps=300)
    prompt = tokenizer("User: Explain KV cache briefly.\nAssistant:", return_tensors="pt").input_ids.to(DEVICE)
    print(tokenizer.decode(model.generate(prompt)[0], skip_special_tokens=True))
