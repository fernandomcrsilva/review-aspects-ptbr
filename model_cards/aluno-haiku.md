---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: cc-by-nc-sa-4.0
language: [pt]
pipeline_tag: text-generation
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
tags: [lora, aspect-based-sentiment-analysis, portuguese, distillation, e-commerce, rocm]
---

# Qwen3-1.7B com LoRA: sentimento por aspecto em avaliações PT-BR (professor Claude Haiku)

Adaptador LoRA para o [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). Ele lê uma avaliação de compra em português e devolve um JSON com o sentimento de cinco aspectos: `entrega`, `produto`, `atendimento`, `preco` e `embalagem`. Cada aspecto recebe `positivo`, `negativo`, `neutro` ou `ausente`.

Foi destilado de rótulos do **Claude Haiku** em 9.000 avaliações da Olist e treinado numa Radeon RX 7600 de 8 GB, com ROCm. É um dos dois braços de um experimento que compara um professor aberto e local com um professor pago via API. O [código e os resultados completos](https://github.com/fernandomcrsilva/review-aspects-ptbr) estão no GitHub.

## Resultados no conjunto ouro público (329 avaliações)

| Métrica | Valor |
|---|---|
| F1-macro (média do F1-macro de cada aspecto) | 0,784 |
| Recall "pede ação" (algum aspecto negativo) | 98,5% |
| JSON válido | 100% |
| Avaliações/s numa RX 7600, batch 16 | 6,6 |

O outro braço (professor Qwen3-8B) e os baselines estão no [repositório](https://github.com/fernandomcrsilva/review-aspects-ptbr#resultados). O ouro é o consenso de dois anotadores LLM, um deles o Claude Opus, e não foi anotado por pessoas. As limitações estão descritas lá.

## Como usar

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "FMendonca/qwen3-1.7b-aspectos-olist-haiku")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

review = "Produto ótimo, mas chegou com uma semana de atraso e a caixa amassada."
prompt = "Aspectos (entrega, produto, atendimento, preco, embalagem) desta avaliação em JSON:\n" + review
enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                              enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# {"entrega": "negativo", "produto": "positivo", "atendimento": "ausente", "preco": "ausente", "embalagem": "negativo"}
```

## Treino

- LoRA com r=16, alpha=32 e dropout 0,05, em todas as projeções de atenção e MLP.
- 2 épocas, lr 2e-4 com decaimento cosseno, batch efetivo 16, bf16.
- Loss só no JSON da resposta.
- 44 minutos numa RX 7600 (ROCm 7.2), com pico de 4,3 GB de VRAM.

## Licença

CC BY-NC-SA 4.0, herdada dos dados da Olist e dos rótulos, então o uso é só não comercial. O modelo base, Qwen3-1.7B, é Apache 2.0.
