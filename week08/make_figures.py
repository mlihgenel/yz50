"""Hafta 8 README figürleri. Sayılar ablation çıktısından ve samples/*.json'dan gelir.
    python make_figures.py   ->  figures/*.png
"""
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = 'figures'

# dataviz referans paleti (light), validate_palette.js ile kontrol edildi
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
BLUE_LIGHT = '#cde2fb'

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK2,
    'xtick.color': MUTED,
    'ytick.color': INK2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 160,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.25,
})


def title(fig, main, sub):
    fig.text(0.0, 1.0, main, fontsize=15, fontweight='bold', color=INK, ha='left', va='bottom',
             transform=fig.axes[0].transAxes if False else fig.transFigure)
    fig.text(0.0, 0.965, sub, fontsize=10.5, color=INK2, ha='left', va='top', transform=fig.transFigure)


def load(name):
    with open(f'samples/{name}.json', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Görev 1-2: ablation merdiveni
# ---------------------------------------------------------------------------
def fig_ablation():
    rows = [  # (etiket, val, param, ana yol mu)
        ('Bigram', 2.472, 4_225, True),
        ('Tek head (32)', 2.370, 7_553, True),
        ('Multi-head (4×8)', 2.218, 7_553, True),
        ('+ FeedForward', 2.122, 15_905, True),
        ('3 blok, residual yok', 2.164, 38_753, False),
        ('3 blok + residual', 2.016, 41_921, True),
        ('+ LayerNorm', 2.019, 42_369, True),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, v, p, main) in zip(ys, rows):
        c = BLUE if main else MUTED
        ax.hlines(y, 1.95, v, color=c, lw=2, alpha=0.35 if main else 0.5)
        ax.plot(v, y, 'o', ms=9, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.text(v + 0.012, y, f'{v:.3f}', va='center', fontsize=10.5, color=INK, fontweight='bold')
        ax.text(2.6, y, f'{p:,} param', va='center', ha='left', fontsize=9.5, color=MUTED, clip_on=False)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(1.95, 2.56)
    ax.set_xlabel('val loss (düşük = iyi)')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.annotate('derinlik tek başına\nişe yaramadı', xy=(2.215, ys[4]), xytext=(2.30, ys[4]),
                fontsize=9.5, color=INK2, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    ax.annotate('aynı param, sadece\nbölünme: −0.15', xy=(2.27, ys[2]), xytext=(2.30, ys[2]),
                fontsize=9.5, color=INK2, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    fig.text(0.0, 1.06, 'Görev 1–2: Transformer bloğunu parça parça kurmak', fontsize=15,
             fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.01, 'Tiny Shakespeare · n_embd 32 · block 8 · 10k adım · CPU — her satır bir öncekinden tek bir şeyle ayrılıyor',
             fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/01_ablation.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Görev 3: büyütme — train/val dumbbell
# ---------------------------------------------------------------------------
def fig_scaling():
    names = [('shakespeare_small', 'small'), ('shakespeare_m1', 'm1'), ('shakespeare_video', 'video')]
    rs = [(lab, load(n)) for n, lab in names]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ys = [2, 1, 0]
    for y, (lab, r) in zip(ys, rs):
        ax.hlines(y, r['train'], r['val'], color=AXIS, lw=3, zorder=1)
        ax.plot(r['train'], y, 'o', ms=10, color=AQUA, mec=SURFACE, mew=2, zorder=3)
        ax.plot(r['val'], y, 'o', ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(r['train'] - 0.03, y, f"{r['train']:.3f}", ha='right', va='center', fontsize=10, color=INK2)
        ax.text(r['val'] + 0.03, y, f"{r['val']:.3f}", ha='left', va='center', fontsize=10.5,
                color=INK, fontweight='bold')
        ax.text((r['train'] + r['val']) / 2, y + 0.22, f"fark {r['val'] - r['train']:.2f}",
                ha='center', fontsize=9, color=MUTED)
        c = r['config']
        dev = 'M1 CPU' if r['device'] == 'cpu' else 'Colab T4'
        ax.text(2.42, y, f"{r['params']/1e6:.2f}M param · {r['time']/60:.0f} dk · {dev}\n"
                f"C{c['n_embd']} · {c['n_head']} head · {c['n_layer']} katman · bağlam {c['block_size']}",
                va='center', fontsize=9, color=INK2)
    ax.set_yticks(ys, [lab for lab, _ in rs], fontsize=12, fontweight='bold')
    ax.set_xlim(0.85, 2.4)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel('loss')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.plot([], [], 'o', color=AQUA, ms=9, label='train')
    ax.plot([], [], 'o', color=BLUE, ms=9, label='val')
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1.12))
    fig.text(0.0, 1.10, 'Görev 3: Büyütmek val loss\'u düşürüyor, ama ezberleme de artıyor', fontsize=15,
             fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.04, 'Tiny Shakespeare · m1 ve video\'da dropout 0.2 · video, Karpathy\'nin videosundaki 1.48\'i tekrar ediyor',
             fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/03_scaling.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. LayerNorm vs BatchNorm: hangi eksende normalize
# ---------------------------------------------------------------------------
def fig_ln_bn():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    n_tok, n_ch = 6, 8
    specs = [
        ('LayerNorm', 'row', BLUE, 'her token kendi C ekseninde\n(test anında da aynı hesap)'),
        ('BatchNorm1d (hafta 6)', 'col', ORANGE, 'her kanal B·T boyunca\n(testte batch yok → running_mean)'),
    ]
    for ax, (name, mode, col, note) in zip(axes, specs):
        for i in range(n_tok):
            for j in range(n_ch):
                hit = (mode == 'row' and i == 2) or (mode == 'col' and j == 3)
                ax.add_patch(Rectangle((j, n_tok - 1 - i), 0.9, 0.9, color=col if hit else GRID,
                                       alpha=0.9 if hit else 1, lw=0))
        ax.set_xlim(-0.6, n_ch + 0.2)
        ax.set_ylim(-1.9, n_tok + 0.9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.text(n_ch / 2, n_tok + 0.35, 'C (kanal) →', ha='center', fontsize=9.5, color=MUTED)
        ax.text(-0.35, n_tok / 2, 'B·T (token) →', ha='center', va='center', rotation=90, fontsize=9.5, color=MUTED)
        ax.text(n_ch / 2, -0.7, name, ha='center', fontsize=12.5, fontweight='bold', color=INK)
        ax.text(n_ch / 2, -1.75, note, ha='center', fontsize=9.5, color=INK2)
    fig.text(0.02, 1.02, 'LayerNorm ile BatchNorm: aynı tensör, farklı eksen', fontsize=15,
             fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.02, 0.97, 'Renkli hücreler: ortalama ve varyansı birlikte hesaplananlar',
             fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/02_layernorm_vs_batchnorm.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Görev 4: Türkçede fazladan gelen karakterler
# ---------------------------------------------------------------------------
def char_group(ch):
    if ch in 'çğıİöşüÇÖÜŞĞ':
        return 'Türkçe harf'
    if ch in 'âîûÂÎÛ':
        return 'Şapkalı (Osmanlıca)'
    if ch in '’‘“”«»–—…':
        return 'Tipografi'
    return 'Rakam ve işaret'


def fig_vocab():
    en = set(open('input.txt', encoding='utf-8').read())
    tr_text = open('turkce.txt', encoding='utf-8').read()
    cnt = Counter(tr_text)
    extra = sorted([c for c in cnt if c not in en], key=lambda c: -cnt[c])
    missing = sorted(c for c in en if c not in cnt)
    groups = ['Türkçe harf', 'Şapkalı (Osmanlıca)', 'Tipografi', 'Rakam ve işaret']
    colors = dict(zip(groups, [BLUE, ORANGE, AQUA, VIOLET]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = range(len(extra))
    for x, c in zip(xs, extra):
        ax.bar(x, cnt[c], width=0.8, color=colors[char_group(c)], lw=0)
        ax.text(x, cnt[c] * 1.2, f'{cnt[c]:,}', ha='center', va='bottom', fontsize=7.5, color=INK2,
                rotation=90)
    ax.set_yscale('log')
    ax.set_ylim(0.8, 6e5)
    ax.set_xticks(list(xs), [repr(c)[1:-1] if c.isspace() else c for c in extra], fontsize=12)
    ax.tick_params(axis='x', length=0, colors=INK)
    ax.set_ylabel('kaç kez geçiyor (log)')
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[g], label=g) for g in groups],
              frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.13))
    fig.text(0.0, 1.10, f'Görev 4: Nutuk\'un vocab\'ı {len(cnt)}, Shakespeare\'inki {len(en)}', fontsize=15,
             fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.04, f'Türkçede olup Shakespeare\'de olmayan {len(extra)} karakter. '
             f'Ters yönde {len(missing)} karakter Shakespeare\'de var, Nutuk\'ta yok: '
             f'{" ".join(missing)}', fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/04_turkce_vocab.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Görev 4: val loss neden doğrudan kıyaslanamaz — farklı başlangıç çizgisi
# ---------------------------------------------------------------------------
def fig_baseline():
    en, tr = load('shakespeare_video'), load('turkce_video')
    rows = [('Shakespeare', en), ('Türkçe (Nutuk)', tr)]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ys = [1, 0]
    for y, (lab, r) in zip(ys, rows):
        base = math.log(r['vocab_size'])
        ax.hlines(y, r['val'], base, color=AXIS, lw=3, zorder=1)
        ax.plot(base, y, 's', ms=10, color=MUTED, mec=SURFACE, mew=2, zorder=3)
        ax.plot(r['val'], y, 'o', ms=11, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(base + 0.06, y, f"ln {r['vocab_size']} = {base:.2f}", va='center', fontsize=10, color=INK2)
        ax.text(r['val'] - 0.06, y, f"{r['val']:.3f}", va='center', ha='right', fontsize=10.5,
                color=INK, fontweight='bold')
        ax.text((base + r['val']) / 2, y + 0.2, f"%{(1 - r['val'] / base) * 100:.0f} düştü",
                ha='center', fontsize=9, color=MUTED)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=11.5, fontweight='bold')
    ax.set_xlim(0.9, 5.4)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('loss')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.plot([], [], 's', color=MUTED, ms=9, label='rastgele tahmin (ln vocab)')
    ax.plot([], [], 'o', color=BLUE, ms=9, label='eğitim sonu val')
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1.2))
    fig.text(0.0, 1.13, 'Görev 4: İki val loss aynı cetvelde değil', fontsize=15, fontweight='bold',
             color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.05, 'Aynı model (video config, 5k adım, T4). Başlangıç çizgisi vocab\'a göre değişiyor; '
             'metin ve dil de farklı', fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/05_baseline_karsilastirma.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Ek (b): makale Şekil 1 -> layers.py
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, tc=INK, ls='-', fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fc, ec=ec, lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold' if bold else 'normal')


def fig_paper_map():
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10.6)
    ax.axis('off')
    GRAY_BG = '#f0efec'
    DIFF_BG = '#fde6dc'

    # sol: encoder (bizde yok)
    box(ax, 0.2, 3.2, 2.2, 4.6, '', GRAY_BG, AXIS, ls='--')
    ax.text(1.3, 7.45, 'ENCODER', ha='center', fontsize=10, color=MUTED, fontweight='bold')
    for i, t in enumerate(['Add & Norm', 'Feed Forward', 'Add & Norm', 'Multi-Head\nAttention']):
        box(ax, 0.45, 6.6 - i * 0.85, 1.7, 0.6, t, SURFACE, AXIS, tc=MUTED, fs=8.5)
    ax.text(1.3, 2.75, 'Inputs + embedding', ha='center', fontsize=8.5, color=MUTED)
    ax.text(1.3, 2.3, 'YOK — çeviri yok,\nkaynak cümle yok', ha='center', va='top', fontsize=9.5,
            color=ORANGE, fontweight='bold')

    # orta: decoder kutuları (makale adı), sağ: kod
    stack = [  # (y, makale, kod, renk tipi)
        (9.6, 'Output Probabilities', 'probs → torch.multinomial  (:111-112)', 'same'),
        (8.95, 'Softmax', 'F.cross_entropy (:103) · F.softmax (:111)', 'same'),
        (8.3, 'Linear', 'lm_head  (:140)', 'same'),
        (7.65, '(şekilde yok)', 'ln_f — pre-norm\'un son LayerNorm\'u  (:139)', 'diff'),
        (6.85, 'Add & Norm', 'x = x + ffwd(ln2(x))  pre-norm  (:81)', 'diff'),
        (6.2, 'Feed Forward', 'FeedForward: C→4C→ReLU→C  (:49)', 'same'),
        (5.4, 'Add & Norm', '—', 'gone'),
        (4.75, 'Multi-Head Attention\n(encoder\'dan K,V)', 'yok: cross-attention', 'gone'),
        (3.95, 'Add & Norm', 'x = x + sa(ln1(x))  pre-norm  (:79)', 'diff'),
        (3.3, 'Masked Multi-Head\nAttention', 'MultiHeadAttention + tril mask  (:13, :22, :30)', 'same'),
        (2.3, 'Positional Encoding ⊕', 'pos_emb — öğrenilen, sin/cos değil  (:133, :146)', 'diff'),
        (1.65, 'Output Embedding', 'token_emb  (:132)', 'same'),
        (1.0, 'Outputs (shifted right)', 'get_batch: x, y = x kaydırılmış  (dataset.py)', 'same'),
    ]
    style = {'same': (BLUE_LIGHT, BLUE, INK), 'diff': (DIFF_BG, ORANGE, INK), 'gone': (GRAY_BG, AXIS, MUTED)}
    # N× çerçevesi
    box(ax, 3.0, 3.0, 2.8, 4.25, '', 'none', INK2, ls='-')
    ax.text(2.85, 6.4, 'N×\n(6)', ha='right', va='center', fontsize=11, fontweight='bold', color=INK2)
    for y, paper, code, kind in stack:
        fc, ec, tc = style[kind]
        h = 0.55
        box(ax, 3.2, y, 2.4, h, paper, fc, ec, tc=tc, fs=8.8, bold=(kind != 'gone'))
        ax.plot([5.65, 6.05], [y + h / 2, y + h / 2], color=ec, lw=1.2, ls='--' if kind == 'gone' else '-')
        ax.text(6.1, y + h / 2, code, va='center', fontsize=9.2, color=tc,
                family='monospace' if kind != 'gone' else None)
        if kind == 'gone':
            ax.plot([3.25, 5.55], [y + h / 2, y + h / 2], color=MUTED, lw=1)
    ax.annotate('', xy=(3.2, 5.0), xytext=(2.4, 5.0),
                arrowprops=dict(arrowstyle='->', color=AXIS, lw=1.2, ls='--'))
    ax.text(4.4, 10.35, 'DECODER (makale)', ha='center', fontsize=10, color=INK2, fontweight='bold')
    ax.text(6.1, 10.35, 'layers.py (bizim kod)', fontsize=10, color=INK2, fontweight='bold')
    # lejant
    for i, (k, lab) in enumerate([('same', 'makaleyle aynı'), ('diff', 'farklı yaptık'), ('gone', 'bizde yok')]):
        fc, ec, _ = style[k]
        box(ax, 6.1 + i * 1.65, 0.15, 0.35, 0.3, '', fc, ec)
        ax.text(6.55 + i * 1.65, 0.3, lab, va='center', fontsize=9, color=INK2)
    fig.text(0.02, 0.99, 'Ek (b): "Attention Is All You Need" Şekil 1 → bizim GPT', fontsize=15,
             fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.02, 0.965, 'GPT = decoder\'ın cross-attention\'sız hali. Satır numaraları layers.py\'den.',
             fontsize=10, color=INK2, transform=fig.transFigure)
    fig.savefig(f'{OUT}/06_makale_sekil1.png')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_ablation()
    fig_ln_bn()
    fig_scaling()
    fig_vocab()
    fig_baseline()
    fig_paper_map()
    print(sorted(os.listdir(OUT)))
