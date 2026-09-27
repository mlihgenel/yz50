"""Model mimarisini çizer: katman yığını + bağlamın birleşme ağacı.

Kullanım:
    python arch_viz.py                      # main.py'deki modeli çizer
    python arch_viz.py turkish_wavenet.py   # başka bir betik
    python arch_viz.py --show               # kaydettikten sonra pencerede de aç

Modeli elle tarif etmiyoruz: betiği (main.py) satır satır çalıştırıp `model = ...` atamasına
gelince duruyoruz, yani eğitim döngüsüne hiç girilmiyor. Böylece main.py'de bir katman
eklenince/çıkarılınca ya da EMB_DIM, HIDDEN_SIZE, DROPOUT_P değişince bu betiği tekrar
çalıştırmak yeter. Şekiller, modelden tek bir batch geçirilip her katmanın `.out`'undan okunuyor.

Çıktı: plots/arch_<betik>_<ARCH>.png (her koşuda üzerine yazılır).
"""
import ast
import os
import sys

import torch
import matplotlib
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import FancyBboxPatch  # type: ignore

# Arkada eğitim koşuyor olabilir; onun CPU'sunu çalmayalım.
torch.set_num_threads(1)

COLORS = {
    "Embedding":          "#8ecae6",
    "Flatten":            "#e9ecef",
    "FlattenConsecutive": "#e9ecef",
    "Linear":             "#ffb703",
    "BatchNorm1d":        "#90be6d",
    "Tanh":               "#cdb4db",
    "Dropout":            "#f28482",
}


def load_model(script):
    """Betiği `model` tanımlanana kadar çalıştırır, namespace'i döndürür."""
    with open(script, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=script)

    ns = {"__name__": "arch_viz_exec"}
    for node in tree.body:
        # Çizim modülüne gerek yok (matplotlib penceresi açmasın).
        if isinstance(node, ast.ImportFrom) and node.module == "viz":
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), ns)
        if "model" in ns:
            return ns
    raise RuntimeError(f"{script} içinde `model = ...` bulunamadı")


def describe(layer):
    name = type(layer).__name__
    if name == "Embedding":
        v, d = layer.weight.shape
        return f"Embedding  {v}×{d}"
    if name == "Linear":
        fi, fo = layer.weight.shape
        return f"Linear  {fi} → {fo}" + ("" if layer.bias is not None else "  (bias yok)")
    if name == "BatchNorm1d":
        return f"BatchNorm1d({layer.bngain.shape[0]})"
    if name == "FlattenConsecutive":
        return f"FlattenConsecutive({layer.n})"
    if name == "Dropout":
        return f"Dropout(p={layer.p})" + ("  — etkisiz" if layer.p == 0 else "")
    return name


def group_layers(layers):
    """Katmanları giriş / gizli katman k / çıkış bloklarına ayırır.

    Kural: yeni blok bir Flatten'la ya da (önceki blokta zaten Linear varken) yeni bir Linear'la başlar.
    """
    groups = []
    for i, layer in enumerate(layers):
        name = type(layer).__name__
        cur = groups[-1] if groups else None
        cur_names = [type(layers[j]).__name__ for j in cur] if cur else []
        starts_new = (
            cur is None
            or name == "Embedding"
            or "Embedding" in cur_names
            or (name in ("Flatten", "FlattenConsecutive") and cur_names)
            or (name == "Linear" and "Linear" in cur_names)
        )
        if starts_new:
            groups.append([i])
        else:
            cur.append(i)

    labels, k = [], 0
    for gi, g in enumerate(groups):
        names = [type(layers[j]).__name__ for j in g]
        if "Embedding" in names:
            labels.append("giriş")
        elif gi == len(groups) - 1 and "Tanh" not in names:
            labels.append("çıkış")
        else:
            k += 1
            labels.append(f"gizli katman {k}")
    return groups, labels


def fmt_shape(t):
    return "(B, " + ", ".join(str(s) for s in t.shape[1:]) + ")"


def draw_stack(ax, model, x_shape):
    layers = model.layers
    groups, labels = group_layers(layers)

    box_h, gap, group_gap = 0.8, 0.25, 0.6
    y = 0.0
    ys = {}
    group_spans = []
    for g in groups:
        top = y
        for j in g:
            ys[j] = y
            y -= box_h + gap
        group_spans.append((top + 0.2, y + gap - 0.2))
        y -= group_gap

    # Giriş (indisler)
    in_y = box_h + gap + 0.3
    ax.text(0.5, in_y + box_h / 2, f"girdi: {BLOCK_SIZE_TXT[0]} harf indeksi   (B, {x_shape[1]})",
            ha="center", va="center", fontsize=10, style="italic")
    ax.annotate("", xy=(0.5, box_h + 0.05), xytext=(0.5, in_y + 0.1),
                arrowprops=dict(arrowstyle="->", color="#555"))

    # Blok arka planları
    for (top, bottom), label in zip(group_spans, labels):
        bg = FancyBboxPatch((-0.08, bottom), 1.16, top - bottom + box_h,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="#f8f9fa", ec="#adb5bd", lw=1, ls="--")
        ax.add_patch(bg)
        ax.text(-0.12, bottom + (top - bottom + box_h) / 2, label, ha="right", va="center",
                fontsize=10, fontweight="bold", color="#495057")

    for j, layer in enumerate(layers):
        name = type(layer).__name__
        yy = ys[j]
        faded = name == "Dropout" and layer.p == 0
        box = FancyBboxPatch((0.1, yy), 0.8, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             fc=COLORS.get(name, "#dee2e6"), ec="#343a40",
                             lw=1, alpha=0.35 if faded else 1.0, ls=":" if faded else "-")
        ax.add_patch(box)
        ax.text(0.5, yy + box_h / 2, describe(layer), ha="center", va="center", fontsize=10,
                color="#868e96" if faded else "black")

        n_params = sum(p.nelement() for p in layer.parameters())
        if n_params:
            ax.text(0.08, yy + box_h / 2, f"{n_params:,} p", ha="right", va="center",
                    fontsize=8, color="#6c757d")
        ax.text(1.12, yy + box_h / 2, fmt_shape(layer.out), ha="left", va="center",
                fontsize=9, family="monospace", color="#1d3557")

        # Sonraki katmana ok
        if j + 1 < len(layers):
            ax.annotate("", xy=(0.5, ys[j + 1] + box_h + 0.02), xytext=(0.5, yy - 0.02),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    last_y = ys[len(layers) - 1]
    ax.annotate("", xy=(0.5, last_y - 0.9), xytext=(0.5, last_y - 0.02),
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.text(0.5, last_y - 1.2, "softmax → sonraki harfin olasılıkları", ha="center",
            va="center", fontsize=10, style="italic")

    ax.text(1.12, box_h + gap + 0.3 + box_h / 2, "çıktı şekli", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#1d3557")
    ax.set_xlim(-0.9, 1.65)
    ax.set_ylim(last_y - 1.7, in_y + box_h + 0.3)
    ax.axis("off")
    ax.set_title("Katmanlar", fontsize=12, fontweight="bold")


def context_levels(model, block_size):
    """Her Flatten adımından sonra kaç 'zaman' konumu kaldığını çıkarır: örn. [8, 4, 2, 1]."""
    levels = [block_size]
    for layer in model.layers:
        if type(layer).__name__ in ("Flatten", "FlattenConsecutive"):
            levels.append(layer.out.shape[1] if layer.out.ndim == 3 else 1)
    return levels


def draw_tree(ax, levels, chars, target):
    n0 = levels[0]
    xs_prev = None
    level_colors = ["#8ecae6", "#ffb703", "#fb8500", "#90be6d", "#cdb4db", "#f28482"]
    for li, n in enumerate(levels):
        y = -li * 1.5
        # Her üst düğüm, altındaki grubun ortasına oturur.
        width = n0 / n
        xs = [(k + 0.5) * width for k in range(n)]
        if xs_prev is not None:
            per = len(xs_prev) // n
            for k, x in enumerate(xs):
                for c in range(k * per, (k + 1) * per):
                    ax.plot([xs_prev[c], x], [y + 1.5 - 0.25, y + 0.25], color="#adb5bd", lw=1.2, zorder=1)
        color = level_colors[li % len(level_colors)]
        for k, x in enumerate(xs):
            ax.add_patch(plt.Circle((x, y), 0.25, fc=color, ec="#343a40", zorder=2))
            if li == 0:
                ax.text(x, y, chars[k], ha="center", va="center", fontsize=11,
                        family="monospace", fontweight="bold", zorder=3)
        side = "harf" if li == 0 else f"{li}. birleşme"
        ax.text(n0 + 0.3, y, f"{side}: {n} konum", ha="left", va="center", fontsize=9, color="#495057")
        xs_prev = xs

    y_end = -(len(levels) - 1) * 1.5
    ax.annotate("", xy=(n0 / 2, y_end - 1.1), xytext=(n0 / 2, y_end - 0.3),
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.text(n0 / 2, y_end - 1.45, f"hedef harf: '{target}'", ha="center", va="center",
            fontsize=11, family="monospace", fontweight="bold")

    ax.set_xlim(-0.3, n0 + 2.8)
    ax.set_ylim(y_end - 2.0, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bağlam nasıl birleşiyor", fontsize=12, fontweight="bold")


BLOCK_SIZE_TXT = [0]  # draw_stack'e metin için


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    script = args[0] if args else "main.py"

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    sys.path.insert(0, here)
    if not show:
        matplotlib.use("Agg")

    ns = load_model(script)
    model = ns["model"]
    block_size = ns["BLOCK_SIZE"]
    BLOCK_SIZE_TXT[0] = block_size

    # Örnek bağlam: başında en az '.' olan train satırı (ağaçta gerçek harfler görünsün).
    X, Y = ns.get("X_train"), ns.get("Y_train")
    if X is not None:
        idx = (X[:5000] != 0).sum(1).argmax().item()
        xb = X[idx:idx + 32]
        itos = ns["itos"]
        chars = [itos[i] for i in X[idx].tolist()]
        target = itos[Y[idx].item()]
    else:
        xb = torch.zeros((32, block_size), dtype=torch.long)
        chars, target = ["·"] * block_size, "?"

    with torch.no_grad():
        model(xb)   # her katmanın .out'u dolsun (ilk batch şekilleri)

    levels = context_levels(model, block_size)
    n_layers = len(model.layers)
    total = sum(p.nelement() for p in model.parameters())

    fig = plt.figure(figsize=(15, max(7, 0.62 * n_layers + 3)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    draw_stack(fig.add_subplot(gs[0]), model, xb.shape)
    draw_tree(fig.add_subplot(gs[1]), levels, chars, target)

    cfg = [f"ARCH={ns.get('ARCH', '?')}", f"BLOCK_SIZE={block_size}"]
    for k in ("EMB_DIM", "HIDDEN_SIZE", "DROPOUT_P", "LR_SCHEDULE"):
        if k in ns:
            cfg.append(f"{k}={ns[k]}")
    fig.suptitle(f"{script}  —  {total:,} parametre\n" + "   ".join(cfg), fontsize=12)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    stem = os.path.splitext(os.path.basename(script))[0]
    out = f"plots/arch_{stem}_{ns.get('ARCH', 'model')}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"kaydedildi: {out}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
