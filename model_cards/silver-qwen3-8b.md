---
license: cc-by-nc-sa-4.0
language: [pt]
task_categories: [text-classification]
tags: [aspect-based-sentiment-analysis, e-commerce, olist, llm-annotation, portuguese]
size_categories: [1K<n<10K]
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# Rótulos de aspectos da Olist gerados por um professor aberto e local (Qwen3-8B)

Rótulos para 10.000 avaliações da Olist em português: 9.000 de treino e 1.000 de validação. Cada avaliação tem o sentimento de cinco aspectos (`entrega`, `produto`, `atendimento`, `preco` e `embalagem`), rotulados como `positivo`, `negativo`, `neutro` ou `ausente`.

Os rótulos foram gerados pelo **Qwen3-8B** (Q4, Ollama com Vulkan numa RX 7600), com a saída JSON restrita ao esquema e o modo de raciocínio desligado. São as mesmas avaliações e as mesmas definições de rótulo dos rótulos do Claude Haiku em [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr), o que torna os dois conjuntos diretamente comparáveis.

Na validação, a concordância com os rótulos do Haiku (kappa de Cohen) é de 0,83 em entrega, 0,74 em produto, 0,69 em atendimento, 0,71 em preço e 0,72 em embalagem.

São rótulos gerados por LLM, não anotação humana. O código e os resultados estão em https://github.com/fernandomcrsilva/review-aspects-ptbr. A licença é CC BY-NC-SA 4.0, herdada do dataset da Olist.
