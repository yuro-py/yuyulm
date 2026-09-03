"""Tiny chat LM: raw PyTorch training/inference, designed for a 4 GB RTX 2050."""
import math
from types import SimpleNamespace
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


def Cfg(vocab=32000, dim=512, layers=10, heads=8, ff=1408, max_seq_len=1024, dropout=0.0):
    # [FUNCTIONAL DATA] was @dataclass class; now a plain factory returning a
    # SimpleNamespace struct with no methods — just data, passed explicitly to functions.
    return SimpleNamespace(vocab=vocab, dim=dim, layers=layers, heads=heads, ff=ff, max_seq_len=max_seq_len, dropout=dropout)


def rmsnorm(x, weight, eps=1e-6):
    # x is [batch, sequence, hidden]; weight is one learned scale per hidden value.
    # [INFERENCE FOCUS] RMSNorm avoids LayerNorm's mean/variance bookkeeping and
    # uses one reduction plus elementwise operations, reducing memory traffic.
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * weight


def rope(x, start):
    # x is [batch, heads, sequence, head_dimension].
    # [INFERENCE FOCUS] Complex numbers express the paired 2-D rotations compactly;
    # the angles are generated on the fly and cached nowhere, keeping the model small.
    b, h, t, d = x.shape
    half = d // 2
    pos = torch.arange(start, start + t, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    z = torch.view_as_complex(x.float().reshape(b, h, t, half, 2))
    rot = torch.polar(torch.ones_like(pos[:, None]), pos[:, None] * inv[None, :])
    return torch.view_as_real(z * rot[None, None]).flatten(-2).to(x.dtype)


def make_attention(c):
    # [FUNCTIONAL] was Attention(nn.Module).__init__; now a factory returning a
    # plain dict of tensors/modules. No hidden self state — caller threads the dict explicitly.
    # pack heads and head_dim into dict for reshaping; two linear weights for QKV and out.
    return {
        "h": c.heads,
        "d": c.dim // c.heads,
        "qkv": nn.Linear(c.dim, 3 * c.dim, bias=False),
        "out": nn.Linear(c.dim, c.dim, bias=False),
    }


def attention_forward(attn, x, past=None):
    # [FUNCTIONAL] was Attention.forward(self, x, past); now attention_forward(attn, x, past)
    # with all state passed in/out explicitly — no self.
    # Read the three useful dimensions from the input tensor.
    b, t, _ = x.shape
    h, d = attn["h"], attn["d"]
    q, k, v = attn["qkv"](x).chunk(3, -1)
    # One projection is cheaper and simpler than three separate Linear layers.
    q = q.view(b, t, h, d).transpose(1, 2)
    # Change [B,T,H,D] into [B,H,T,D], the convenient attention layout.
    k = k.view(b, t, h, d).transpose(1, 2)
    v = v.view(b, t, h, d).transpose(1, 2)
    old = 0 if past is None else past[0].size(2)
    # old tells RoPE where these new tokens occur in the full sequence.
    q, k = rope(q, old), rope(k, old)
    # [INFERENCE FOCUS] This explicit concat is a memory copy. Production
    # engines use paged/ring KV storage; it is visible here for learning.
    if past is not None:
        # Append old keys/values so each new token can see the whole history.
        k = torch.cat((past[0], k), dim=2)
        v = torch.cat((past[1], v), dim=2)
    # [INFERENCE FOCUS] Truncating bounds KV memory and attention work on 4 GB.
    if k.size(2) > MAX_SEQ_LEN:
        k, v = k[:, :, -MAX_SEQ_LEN:], v[:, :, -MAX_SEQ_LEN:]
        old = k.size(2) - t
    # [INFERENCE FOCUS] Explicit attention exposes the [Q,K] score matrix;
    # its memory and compute grow quadratically with context length.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if old:
        # Cached keys are visible; future keys among the current tokens are not.
        allow = torch.ones(t, k.size(2), device=x.device, dtype=torch.bool)
        allow = torch.tril(allow, diagonal=old)
        allow = allow[None, None]
        scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
    else:
        # A prompt token cannot look ahead at later prompt tokens.
        future = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(future[None, None], torch.finfo(scores.dtype).min)
    # Softmax converts each row of scores into attention probabilities.
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    # Each output is the probability-weighted sum of value vectors.
    y = probs @ v
    # Return to [B,T,H*D] and apply the output projection.
    return attn["out"](y.transpose(1, 2).reshape(b, t, -1)), (k, v)


def make_block(c):
    # [FUNCTIONAL] was Block(nn.Module).__init__; now a factory for one transformer block.
    # dict holds two RMSNorm scales + attn dict + three linear weights; no class needed.
    return {
        "n1": nn.Parameter(torch.ones(c.dim)),
        "n2": nn.Parameter(torch.ones(c.dim)),
        "attn": make_attention(c),
        "w1": nn.Linear(c.dim, c.ff, bias=False),
        "w2": nn.Linear(c.ff, c.dim, bias=False),
        "w3": nn.Linear(c.dim, c.ff, bias=False),
    }


def block_forward(block, x, past=None):
    # [FUNCTIONAL] was Block.forward(self, x, past); now block_forward(block, x, past)
    # Normalize before attention, then add its residual output to x.
    a, cache = attention_forward(block["attn"], rmsnorm(x, block["n1"]), past)
    x = x + a
    # [INFERENCE FOCUS] SwiGLU is silu(xW_gate) * (xW_up), with no GELU module.
    z = rmsnorm(x, block["n2"])
    # Compute the gated feed-forward network and add its residual output.
    x = x + block["w2"](F.silu(block["w1"](z)) * block["w3"](z))
    return x, cache


def make_model(c):
    # [FUNCTIONAL] was YuYuLM(nn.Module).__init__; now a factory returning a plain dict.
    # embedding table turns integer token IDs into dense vectors.
    tok = nn.Embedding(c.vocab, c.dim)
    blocks = [make_block(c) for _ in range(c.layers)]
    norm = nn.Parameter(torch.ones(c.dim))
    model = {"cfg": c, "tok": tok, "blocks": blocks, "norm": norm}
    # weight tying: lm head reuses embedding weight (same tensor object, not a copy)
    model["lm_weight"] = tok.weight
    return model


def model_parameters(model):
    # [FUNCTIONAL] was model.parameters(); now explicit generator over dict-held tensors.
    # lm_weight is tied to tok.weight, so yield only once.
    yield model["tok"].weight
    yield model["norm"]
    for block in model["blocks"]:
        yield block["n1"]
        yield block["n2"]
        yield from block["attn"]["qkv"].parameters()
        yield from block["attn"]["out"].parameters()
        yield from block["w1"].parameters()
        yield from block["w2"].parameters()
        yield from block["w3"].parameters()


def model_to(model, device):
    # [FUNCTIONAL] was model.to(device); now explicit move of each leaf tensor/module.
    model["tok"] = model["tok"].to(device)
    # nn.Parameter .to() returns Tensor, so re-wrap to keep Parameter semantics
    model["norm"] = nn.Parameter(model["norm"].to(device))
    model["lm_weight"] = model["tok"].weight  # re-tie after move
    for block in model["blocks"]:
        block["n1"] = nn.Parameter(block["n1"].to(device))
        block["n2"] = nn.Parameter(block["n2"].to(device))
        block["attn"]["qkv"] = block["attn"]["qkv"].to(device)
        block["attn"]["out"] = block["attn"]["out"].to(device)
        block["w1"] = block["w1"].to(device)
        block["w2"] = block["w2"].to(device)
        block["w3"] = block["w3"].to(device)
    return model


def model_forward(model, ids, past=None, targets=None):
    # [FUNCTIONAL] was YuYuLM.forward(self, ids, past, targets); now model_forward(model, ids, ...)
    # ids is [batch, tokens], containing vocabulary indices.
    x = model["tok"](ids)
    # x is now [batch, tokens, hidden].
    new = []
    for i, block in enumerate(model["blocks"]):
        # Give each block its own layer's cached keys and values.
        x, cache = block_forward(block, x, None if past is None else past[i])
        new.append(cache)
    logits = F.linear(rmsnorm(x, model["norm"]), model["lm_weight"])
    # logits scores every vocabulary token at every input position.
    loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, tuple(new), loss


@torch.no_grad()
def model_generate(model, ids, max_new=80, temperature=0.8, top_k=40):
    # [FUNCTIONAL] was YuYuLM.generate(self, ids, ...); now model_generate(model, ids, ...)
    # Generation needs no gradients, so decorator/context keeps VRAM lower.
    # [FUNCTIONAL] no self.eval() — functional model has no train/eval mode; dropout is 0 so no-op.
    past = None
    for _ in range(max_new):
        # [INFERENCE FOCUS] First pass consumes the prompt; later passes consume
        # one token and reuse every prior K/V, changing O(T²) prompt work to O(T).
        logits, past, _ = model_forward(model, ids if past is None else ids[:, -1:], past)
        p = logits[:, -1] / max(temperature, 1e-5)
        # Temperature controls randomness: higher values flatten probabilities.
        v, ix = torch.topk(p, min(top_k, p.size(-1)))
        # Restrict sampling to the most likely candidates.
        probs = F.softmax(v, -1)
        ids = torch.cat((ids, ix.gather(-1, torch.multinomial(probs, 1))), 1)
    return ids


def train_one_epoch(model, tokenizer, steps=300, batch=2, seq=512, lr=3e-4):
    # This is a deliberately small training smoke test, not a full pre-training run.
    # [BOILERPLATE] SmolTalk provides conversational text; a small bounded slice
    # keeps this smoke-training run feasible on a 4 GB card.
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    opt = AdamW(model_parameters(model), lr=lr, weight_decay=0.1)
    it = iter(ds)
    # [FUNCTIONAL] no model.train() — functional dict has no module mode; dropout is 0 anyway.
    for step in range(steps):
        # Build one small batch from the streaming conversation dataset.
        texts = []
        for _ in range(batch):
            row = next(it)
            # Fetch only one example; the complete dataset is never loaded in RAM.
            msgs = row.get("messages", [])
            texts.append("\n".join(str(m.get("content", "")) for m in msgs))
        ids = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq).input_ids.to(DEVICE)
        # Tokenize on CPU, then move only this batch to the selected device.
        if ids.size(1) < 2:
            continue
        _, _, loss = model_forward(model, ids[:, :-1], targets=ids[:, 1:])
        # Shift inputs and labels so each token learns to predict the next token.
        # Clear gradients, compute gradients, then update parameters.
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"step {step}: loss {loss.item():.3f}")


if __name__ == "__main__":
    # This guard runs the demo only when the file is launched directly.
    # [AUTOMATION] Later wrap model_forward with torch.compile() to fuse
    # RMSNorm/SwiGLU kernels and reduce Python dispatch overhead.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    cfg = Cfg(vocab=len(tokenizer), max_seq_len=MAX_SEQ_LEN)
    model = make_model(cfg)
    model = model_to(model, DEVICE)
    print(f"parameters: {sum(p.numel() for p in model_parameters(model)) / 1e6:.1f}M on {DEVICE}")
    # [AUTOMATION] Replace steps with a checkpointed scheduler and save every N
    # steps for unattended overnight training; this dummy run intentionally stays small.
    train_one_epoch(model, tokenizer, steps=300)
    prompt = tokenizer("User: Hi.\nAssistant:", return_tensors="pt").input_ids.to(DEVICE)
    print(tokenizer.decode(model_generate(model, prompt)[0], skip_special_tokens=True))
