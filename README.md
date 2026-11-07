# Professor aberto ou pago? Destilação de LLM para análise de avaliações da Olist numa GPU AMD de 8 GB

Quando se destila um LLM grande num modelo pequeno, quem gera os rótulos de treino faz diferença. Este projeto mede essa diferença. O mesmo modelo, um Qwen3-1.7B com LoRA, foi treinado duas vezes sobre as mesmas 9 mil avaliações da Olist. Numa delas, os rótulos vieram do Claude Haiku, que é pago e roda na nuvem. Na outra, vieram do Qwen3-8B, aberto e gratuito, rodando em casa numa Radeon RX 7600 de 8 GB. No fim, os dois alunos foram avaliados no mesmo conjunto de teste público.

Para cada avaliação, o modelo diz se o cliente falou bem, mal ou de forma neutra de cinco aspectos da compra (entrega, produto, atendimento, preço e embalagem), ou se nem chegou a falar deles. A resposta sai em JSON, pronta para um sistema de atendimento decidir para onde encaminhar o caso.

Os modelos e os rótulos estão no Hugging Face: o [aluno treinado com o Haiku](https://huggingface.co/FMendonca/qwen3-1.7b-aspectos-olist-haiku), o [aluno treinado com o Qwen3-8B](https://huggingface.co/FMendonca/qwen3-1.7b-aspectos-olist-qwen3-8b) e os [10 mil rótulos gerados pelo Qwen3-8B](https://huggingface.co/datasets/FMendonca/olist-aspectos-silver-qwen3-8b).

A história completa do experimento, com o que deu certo e o que surpreendeu, está no artigo [Vale pagar uma API para rotular dados?](artigo/artigo.md).

## Resultados

O conjunto de teste tem 329 avaliações. A métrica é a mesma do [trabalho de referência](https://github.com/renatospessotto/review-intelligence-ptbr): calcula-se o F1 macro de cada aspecto, considerando só as classes que aparecem, e tira-se a média dos cinco. Os intervalos de confiança de 95% vêm de 1.000 reamostragens (bootstrap).

A coluna "pede ação" mede outra coisa: quantas das avaliações com algum aspecto negativo o modelo encontrou. São as avaliações que exigiriam uma resposta da empresa.

| Modelo | F1 macro (IC 95%) | Encontrou "pede ação" | Avaliações por segundo* |
|---|---|---|---|
| **Aluno treinado com rótulos do Haiku** | **0,784** (0,735 a 0,861) | **98,5%** | 6,6 |
| Aluno treinado com rótulos do Qwen3-8B | 0,690 (0,645 a 0,769) | 96,2% | 6,8 |
| TF-IDF com regressão logística, rótulos do Haiku | 0,732 (0,671 a 0,769) | 94,6% | 14,7 (CPU) |
| Professor Qwen3-8B, sem ajuste, via Ollama | 0,709 (0,676 a 0,769) | 96,2% | 0,9 em série; 2,5 com 4 em paralelo |
| TF-IDF com regressão logística, rótulos do Qwen3-8B | 0,619 (0,580 a 0,650) | 95,4% | 17,0 (CPU) |
| Aluno sem ajuste fino | 0,472 (0,434 a 0,499) | 91,5% | 3,7 |

\* Medido num Ryzen 7 5700X com uma RX 7600.

Comparando os dois alunos nas mesmas reamostragens (bootstrap pareado), o treinado com o Haiku fica 0,091 à frente, com intervalo de confiança de +0,007 a +0,160.

F1 macro por aspecto:

| Modelo | Entrega | Produto | Atendimento | Preço | Embalagem |
|---|---|---|---|---|---|
| Aluno treinado com o Haiku | 0,70 | 0,81 | 0,83 | 0,97 | 0,61 |
| Aluno treinado com o Qwen3-8B | 0,70 | 0,70 | 0,74 | 0,72 | 0,60 |
| Professor Qwen3-8B | 0,70 | 0,65 | 0,62 | 0,98 | 0,58 |
| TF-IDF com rótulos do Haiku | 0,66 | 0,69 | 0,61 | 0,94 | 0,75 |

## Conclusões

O aluno não vai muito além do professor. Treinado com os rótulos do Qwen3-8B, ele chegou a 97% do F1 do próprio professor (0,690 contra 0,709), só que cerca de três vezes mais rápido: processa 6,8 avaliações por segundo em lotes de 16, enquanto o professor faz 2,5 com quatro requisições simultâneas. Com o Haiku como professor, o mesmo aluno foi a 0,784.

O Haiku ganhou, mas a vantagem é menor do que parece. O intervalo de confiança da diferença quase encosta no zero. Além disso, o conjunto de teste foi anotado pelo Claude Opus junto com outro modelo, o que pode favorecer rótulos da família Claude. Em entrega, o aspecto que mais importa para uma operação de logística, os dois alunos empatam em 0,70.

Em alguns aspectos o aluno superou o professor. Em atendimento (0,74 contra 0,62) e em produto (0,70 contra 0,65), o aluno do Qwen3-8B foi melhor que o próprio Qwen3-8B. A explicação mais provável é que o professor erra de forma pouco consistente, e o treino com 9 mil exemplos acaba suavizando esses erros. Em preço aconteceu o contrário (0,72 contra 0,98), mas aí pesa o tamanho da amostra: só 16 avaliações do teste falam de preço, e com tão poucos casos cada erro muda muito a nota.

Se o objetivo é só triagem, qualquer um dos dois alunos serve. Ambos encontram entre 96% e 98% das avaliações que pedem ação, e a diferença entre eles aparece no detalhe de cada aspecto.

E tudo isso sem pagar nenhuma API. O Qwen3-8B rotulou as 10 mil avaliações em cerca de 67 minutos pelo Ollama com Vulkan, com o JSON restrito ao esquema. Cada aluno levou 44 minutos para treinar, usando no máximo 4,3 GB de VRAM.

## Como coube numa placa AMD de 8 GB

O PyTorch com ROCm 7.2 reconheceu a RX 7600 (gfx1102) sem nenhum ajuste. Não foi preciso nem o `HSA_OVERRIDE_GFX_VERSION`, que costuma ser necessário em placas fora da lista oficial da AMD. O problema foi a memória: com um LoRA convencional, um lote de 4 sequências de 192 tokens já não cabia nos 8 GB. Três mudanças em [`src/train_lora.py`](src/train_lora.py) resolveram, e hoje um lote de 8 usa no máximo 4,3 GB.

1. **Logits só na resposta.** O vocabulário do Qwen3 tem 151 mil tokens, então os logits da sequência inteira são o maior tensor do treino. A loss passa pelo `lm_head` apenas nas posições do JSON de resposta.
2. **Prompt curto para o aluno.** O professor precisa do guia de anotação completo, com uns 400 tokens. Depois de treinado, o aluno já conhece o formato e recebe uma linha só, com uns 80 tokens. Só essa mudança deixou o treino 3,4 vezes mais rápido, e a inferência também fica mais barata.
3. **Padding em múltiplos de 32.** O ROCm escolhe um kernel para cada formato de tensor. Com menos formatos diferentes, ele perde menos tempo nessa escolha.

## Como reproduzir

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2   # ou a versão para CUDA
pip install -r requirements.txt
python src/get_data.py                                                   # baixa o teste e os rótulos do Haiku
python src/teacher_label.py --model qwen3:8b --out data/silver-qwen3-8b  # precisa do `ollama serve` rodando
python src/train_lora.py --silver data/aspectos/silver --out runs/aluno-haiku --batch 8 --accum 2
python src/train_lora.py --silver data/silver-qwen3-8b --out runs/aluno-qwen3-8b --batch 8 --accum 2
python src/evaluate.py --name aluno-haiku --adapter runs/aluno-haiku     # repita para cada modelo
python src/report.py
```

O notebook [`notebooks/kaggle_treino.ipynb`](notebooks/kaggle_treino.ipynb) refaz o treino do aluno com rótulos do Haiku numa T4 gratuita do Kaggle, em fp32, porque a T4 não tem bf16 nativo. Na [execução pública no Kaggle](https://www.kaggle.com/code/fernandomcrsilva/review-aspects-ptbr-treino), os números ficaram quase iguais aos da RX 7600: 0,791 no aluno (0,784 aqui), 0,732 no TF-IDF e 0,473 no aluno sem ajuste fino.

| Arquivo | Função |
|---|---|
| [`src/aspects.py`](src/aspects.py) | Esquema, prompts, leitura do JSON e métrica |
| [`src/teacher_label.py`](src/teacher_label.py) | Rotulagem pelo professor aberto via Ollama, que pode ser retomada se parar |
| [`src/train_lora.py`](src/train_lora.py) | Treino com LoRA, em ROCm ou CUDA |
| [`src/evaluate.py`](src/evaluate.py) | Avaliação de alunos, professor e TF-IDF no conjunto de teste |
| [`src/report.py`](src/report.py) | Intervalos de confiança e comparação pareada |

## Limitações

- **O conjunto de teste foi anotado por LLMs, não por pessoas.** Ele é o consenso entre o Claude Opus 5.5 e um segundo modelo. Ficaram só as 329 avaliações, de 400, em que os dois concordaram nos cinco aspectos, e casos de consenso costumam ser os mais fáceis.
- **O teste é pequeno.** Preço e embalagem aparecem em apenas 16 e 13 avaliações, então o F1 desses aspectos oscila muito. Vale olhar os intervalos de confiança antes de comparar.
- **Cada aluno foi treinado uma única vez,** com seed 0. Não sei quanto o resultado varia de uma execução para outra.
- **Não ajustei o prompt do professor.** Escrevi o prompt uma vez, a partir do guia de anotação do conjunto de teste e antes de ver qualquer resultado, justamente para não otimizar em cima do teste.
- **São avaliações de e-commerce,** não de delivery de comida.

## Dados, créditos e licenças

- As avaliações vêm do [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), sob CC BY-NC-SA 4.0.
- O conjunto de teste e os rótulos do Claude Haiku são de [RenatoSpessotto/olist-aspectos-reviews-ptbr](https://huggingface.co/datasets/RenatoSpessotto/olist-aspectos-reviews-ptbr), também sob CC BY-NC-SA 4.0. Este projeto reaproveita esses dados e a definição da métrica para medir uma única variável nova: o professor.
- Os modelos usados são o [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), como aluno, e o [Qwen3-8B](https://ollama.com/library/qwen3), como professor, ambos sob Apache 2.0.
- O código deste repositório está sob [MIT](LICENSE). Os rótulos e os adaptadores derivam de dados CC BY-NC-SA 4.0 e herdam essa licença, ou seja, só podem ser usados para fins não comerciais e redistribuídos sob a mesma licença.
