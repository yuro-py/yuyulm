from bench import measure_generate, measure_perplexity, print_report





golden = torch.load("golden_logits.pt")
print(torch.allclose(your_logits, golden, atol=1e-3))
print((your_logits - golden).abs().max())  # how far off, if any
