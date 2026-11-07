"""Gera artigo/f1.png a partir dos números de results/tabela.md."""
import matplotlib.pyplot as plt

HAIKU, QWEN, SEM = "#2a78d6", "#eb6834", "#8a8984"
# (rótulo, F1, IC inferior, IC superior, cor), na ordem do gráfico, de cima para baixo
linhas = [
    ("Aluno Qwen3-1.7B, rótulos do Haiku", 0.784, 0.735, 0.861, HAIKU),
    ("TF-IDF, rótulos do Haiku", 0.732, 0.671, 0.769, HAIKU),
    ("Professor Qwen3-8B, sem ajuste", 0.709, 0.676, 0.769, QWEN),
    ("Aluno Qwen3-1.7B, rótulos do Qwen3-8B", 0.690, 0.645, 0.769, QWEN),
    ("TF-IDF, rótulos do Qwen3-8B", 0.619, 0.580, 0.650, QWEN),
    ("Aluno sem ajuste fino", 0.472, 0.434, 0.499, SEM),
]
virgula = lambda x, casas=2: f"{x:.{casas}f}".replace(".", ",")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
for i, (nome, f1, lo, hi, cor) in enumerate(linhas):
    y = len(linhas) - 1 - i
    ax.plot([lo, hi], [y, y], color=cor, lw=2, solid_capstyle="round")
    ax.plot(f1, y, "o", ms=9, color=cor, mec="#fcfcfb", mew=2)
    ax.text(hi + 0.012, y, virgula(f1, 3), va="center", color="#0b0b0b")
ax.set_yticks(range(len(linhas)), [l[0] for l in reversed(linhas)], color="#0b0b0b")
ax.set_xlim(0.4, 0.92)
ax.xaxis.set_major_formatter(lambda x, _: virgula(x))
ax.tick_params(axis="both", length=0, colors="#52514e")
ax.grid(axis="x", color="#e4e3df", lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("F1 macro no teste, com intervalo de confiança de 95%", color="#52514e")
for cor, nome in [(HAIKU, "Rótulos do Claude Haiku (pago)"), (QWEN, "Rótulos do Qwen3-8B (aberto)"),
                  (SEM, "Sem rótulos")]:
    ax.plot([], [], "o", ms=8, color=cor, label=nome)
fig.legend(loc="upper center", ncol=3, frameon=False, handletextpad=0.3)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("artigo/f1.png", facecolor=fig.get_facecolor())
