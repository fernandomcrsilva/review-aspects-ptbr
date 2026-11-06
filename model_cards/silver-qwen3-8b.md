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

# Olist aspect labels from an open, local teacher (Qwen3-8B)

Labels for 10,000 Brazilian Portuguese Olist reviews (9,000 train and 1,000 validation). Each review has the sentiment of five aspects (`entrega`, `produto`, `atendimento`, `preco`, `embalagem`), labelled `positivo`, `negativo`, `neutro` or `ausente`.

The labels were produced by **Qwen3-8B** (Q4, Ollama on Vulkan, RX 7600), with JSON output constrained to the schema and thinking disabled. These are the same reviews and the same label definitions as the Claude Haiku labels in [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr). That makes the two label sets directly comparable.

On the validation split, agreement with the Haiku labels (Cohen's kappa) is 0.83 for delivery, 0.74 for product, 0.69 for customer service, 0.71 for price and 0.72 for packaging.

These are LLM labels, not human annotations. Code and results: https://github.com/fernandomcrsilva/review-aspects-ptbr. The license is CC BY-NC-SA 4.0, inherited from the Olist dataset.
