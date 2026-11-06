---
base_model: Qwen/Qwen3-1.7B
library_name: peft
license: cc-by-nc-sa-4.0
language: [pt]
pipeline_tag: text-generation
datasets: [RenatoSpessotto/olist-aspectos-reviews-ptbr]
tags: [lora, aspect-based-sentiment-analysis, portuguese, distillation, e-commerce, rocm]
---

# Qwen3-1.7B com LoRA para análise de aspectos em avaliações da Olist (rótulos do Claude Haiku)

Este adaptador LoRA faz o [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) ler uma avaliação de compra em português e dizer como o cliente falou de cinco aspectos: `entrega`, `produto`, `atendimento`, `preco` e `embalagem`. Cada aspecto recebe `positivo`, `negativo`, `neutro` ou `ausente`, e a resposta sai em JSON.

O modelo foi treinado com 9 mil avaliações da Olist rotuladas pelo Claude Haiku, um modelo pago acessado por API. O treino rodou numa Radeon RX 7600 de 8 GB, com ROCm. Ele faz parte de um experimento que compara o que um aluno aprende com um professor aberto e local e com um professor pago. O [código e os resultados completos](https://github.com/fernandomcrsilva/review-aspects-ptbr) estão no GitHub.

## Resultados no conjunto de teste público (329 avaliações)

| Métrica | Valor |
|---|---|
| F1 macro (média dos cinco aspectos) | 0,784 |
| Avaliações com aspecto negativo encontradas | 98,5% |
| Respostas em JSON válido | 100% |
| Avaliações por segundo numa RX 7600, em lotes de 16 | 6,6 |

A comparação com o aluno treinado pelo Qwen3-8B e com os outros modelos está no [repositório](https://github.com/fernandomcrsilva/review-aspects-ptbr#resultados). O conjunto de teste é o consenso de dois LLMs, um deles o Claude Opus, e não foi revisado por pessoas. As limitações estão explicadas lá.

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

O LoRA tem r=16, alpha=32 e dropout de 0,05, aplicado em todas as projeções de atenção e de MLP. Foram 2 épocas com taxa de aprendizado de 2e-4, decaimento cosseno, lote efetivo de 16 e bf16. A loss considera só o JSON da resposta. O treino levou 44 minutos numa RX 7600 com ROCm 7.2 e usou no máximo 4,3 GB de VRAM.

## Licença

CC BY-NC-SA 4.0, herdada dos dados da Olist e dos rótulos, então o uso é só não comercial. O modelo base, Qwen3-1.7B, é Apache 2.0.
