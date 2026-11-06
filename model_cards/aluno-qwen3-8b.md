---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: cc-by-nc-sa-4.0
language: [pt]
pipeline_tag: text-generation
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
tags: [lora, aspect-based-sentiment-analysis, portuguese, distillation, e-commerce, rocm]
---

# Qwen3-1.7B LoRA: aspect sentiment for PT-BR reviews (Qwen3-8B (open, local) teacher)

LoRA adapter for [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). It reads a Brazilian Portuguese purchase review and returns JSON with the sentiment of five aspects: `entrega` (delivery), `produto` (product), `atendimento` (customer service), `preco` (price) and `embalagem` (packaging). Each aspect gets one of `positivo` / `negativo` / `neutro` / `ausente` (positive / negative / neutral / absent).

It was distilled from **Qwen3-8B (open, local)** labels on 9,000 Olist reviews and trained on a Radeon RX 7600 (8 GB, ROCm). It is one of the two arms of an experiment comparing an open, local teacher with a paid API teacher: [code and full results](https://github.com/fernandomcrsilva/review-aspects-ptbr).

## Results on the public gold set (329 reviews)

| Metric | Value |
|---|---|
| Macro-F1 (mean of the per-aspect macro-F1) | 0.690 |
| Recall, "needs action" (any negative aspect) | 96.2% |
| Valid JSON | 100% |
| Reviews/s on an RX 7600, batch 16 | 6.8 |

The other arm (the Claude Haiku teacher) and the baselines are in the [repository](https://github.com/fernandomcrsilva/review-aspects-ptbr#results). The gold set is a consensus of two LLM annotators, one of them Claude Opus, and was not annotated by people. Read the limitations there.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "fernandomcrsilva/<this-repo>")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

review = "Produto ótimo, mas chegou com uma semana de atraso e a caixa amassada."
prompt = "Aspectos (entrega, produto, atendimento, preco, embalagem) desta avaliação em JSON:\n" + review
enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

- LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections.
- 2 epochs, lr 2e-4 with cosine decay, effective batch 16, bf16.
- Loss only on the JSON answer.
- 44 min on an RX 7600 (ROCm 7.2), with a 4.3 GB VRAM peak.

## License

CC BY-NC-SA 4.0, inherited from the Olist data and the labels, so non-commercial use only. The base model, Qwen3-1.7B, is Apache 2.0.
