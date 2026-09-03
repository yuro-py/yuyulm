# bench.py — shared benchmarking utilities. Import this from every step file
# so all your runs measure things identically and stay comparable.
import time
import torch


def reset_gpu_stats():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def measure_generate(model, tok, ids, attn_mask, max_new_tokens=50):
    reset_gpu_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(ids, attention_mask=attn_mask, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    past = out.past_key_values

    next_id = out.logits[:, -1:].argmax(-1)
    generated = [next_id]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        generated.append(next_id)
    torch.cuda.synchronize()
    decode_total_ms = (time.perf_counter() - t0) * 1000

    n_decoded = len(generated) - 1
    decode_ms_per_tok = decode_total_ms / max(n_decoded, 1)
    tokens_per_sec = 1000 / decode_ms_per_tok if decode_ms_per_tok > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

    # DynamicCache stores per-layer tensors in parallel lists, not (k,v) pairs
    kv_cache_mb = 0
    key_list = getattr(past, "key_cache", None)
    value_list = getattr(past, "value_cache", None)
    if key_list is not None and value_list is not None:
        for k in key_list:
            if k is not None and k.numel() > 0:
                kv_cache_mb += k.numel() * k.element_size()
        for v in value_list:
            if v is not None and v.numel() > 0:
                kv_cache_mb += v.numel() * v.element_size()
    kv_cache_mb /= 1e6

    full_ids = torch.cat([ids] + generated, dim=1)

    return {
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms_per_tok": round(decode_ms_per_tok, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "n_tokens_generated": n_decoded + 1,
        "output_ids": full_ids,
    }


def measure_perplexity(model, ids):
    """Perplexity on a fixed sequence — treats ids as both input and target,
    shifted by one. Use the SAME fixed text across every step file so this
    number is directly comparable run to run.
    """
    with torch.no_grad():
        out = model(ids[:, :-1], labels=ids[:, 1:])
    return torch.exp(out.loss).item()


def print_report(tag, metrics):
    print(f"\n--- {tag} ---")
    for k, v in metrics.items():
        if k != "output_ids":
            print(f"{k:22s}: {v}")
