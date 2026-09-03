# run_baseline.py — verifies stock model runs, no custom code touched
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")

msgs = [{"role": "user", "content": "are u stupid?"}]

ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if hasattr(ids, "input_ids"):
    ids = ids.input_ids
ids = ids.to("cuda")
attn_mask = torch.ones_like(ids)

attn_mask = torch.ones_like(ids)
out = model.generate(ids, attention_mask=attn_mask, max_new_tokens=50, pad_token_id=tok.eos_token_id)

print(tok.decode(out[0], skip_special_tokens=True))

with torch.no_grad():
    outputs = model(ids, attention_mask=attn_mask)
    torch.save(outputs.logits, "golden_logits.pt")
