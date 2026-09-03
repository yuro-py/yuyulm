# run_baseline.py — verifies stock model runs, no custom code touched.
# This is your golden reference: never modify this file after it's working.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from bench import measure_generate, measure_perplexity, print_report

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
model.eval()

msgs = [{"role": "user", "content": "are u stupid?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
ids = ids.to("cuda")
attn_mask = torch.ones_like(ids)

metrics = measure_generate(model, tok, ids, attn_mask, max_new_tokens=50)
print(tok.decode(metrics["output_ids"][0], skip_special_tokens=True))
print_report("baseline (HF stock)", metrics)

# fixed held-out text for perplexity — use this EXACT string in every step file
FIXED_TEXT = "The quick brown fox jumps over the lazy dog. Paris is the capital of France."
fixed_ids = tok(FIXED_TEXT, return_tensors="pt").input_ids.to("cuda")
ppl = measure_perplexity(model, fixed_ids)
print(f"perplexity on fixed text: {ppl:.3f}")

# golden logits for exact-match comparison in step B
with torch.no_grad():
    outputs = model(ids, attention_mask=attn_mask)
    torch.save(outputs.logits, "golden_logits.pt")
