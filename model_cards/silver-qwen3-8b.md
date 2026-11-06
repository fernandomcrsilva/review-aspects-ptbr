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

# Aspectos de 10 mil avaliações da Olist rotulados pelo Qwen3-8B

São 10 mil avaliações da Olist em português, 9 mil para treino e mil para validação. Cada uma traz o sentimento do cliente sobre cinco aspectos da compra (`entrega`, `produto`, `atendimento`, `preco` e `embalagem`), marcado como `positivo`, `negativo`, `neutro` ou `ausente`.

Quem rotulou foi o Qwen3-8B, quantizado em 4 bits e rodando localmente pelo Ollama com Vulkan, numa RX 7600. O JSON de saída foi restrito ao esquema, e o modo de raciocínio do modelo ficou desligado. As avaliações e as definições dos rótulos são as mesmas dos rótulos do Claude Haiku em [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr), então os dois conjuntos podem ser comparados linha a linha.

Na validação, a concordância com o Haiku, medida pelo kappa de Cohen, ficou em 0,83 para entrega, 0,74 para produto, 0,69 para atendimento, 0,71 para preço e 0,72 para embalagem.

Os rótulos foram gerados por um LLM e não foram revisados por pessoas. O código e os resultados estão em https://github.com/fernandomcrsilva/review-aspects-ptbr. A licença é CC BY-NC-SA 4.0, herdada do dataset da Olist.
