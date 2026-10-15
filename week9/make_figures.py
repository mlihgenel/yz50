"""Hafta 9 README figürleri. Tokenizer sayıları burada yeniden hesaplanır, ek (b) sayıları
samples/turkce_bpe512_video.json ve train_gpt.ipynb çıktısından gelir.
    python make_figures.py   ->  figures/*.png   (~1 dk)
"""
import json
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import regex as re
import tiktoken
from matplotlib.patches import FancyBboxPatch, Patch

from bpe import get_stats, merge
from regex_tokenizer import GPT4_SPLIT_PATTERN, TR_SPLIT_PATTERN, RegexTokenizer
from compare_tr_en import EN, TR

OUT = 'figures'
DATA = '../week8/turkce.txt'

# hafta 8 ile aynı palet (dataviz referans paleti, light)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
BLUE_LIGHT = '#cde2fb'
ORANGE_LIGHT = '#fbdccf'
AQUA_LIGHT = '#c9eedf'
VIOLET_LIGHT = '#dcd8f7'

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
MONO = 'Menlo' if sys.platform == 'darwin' else 'DejaVu Sans Mono'


def heading(fig, main, sub, y=1.06, gap=0.05):
    fig.text(0.0, y, main, fontsize=15, fontweight='bold', color=INK, transform=fig.transFigure)
    fig.text(0.0, y - gap, sub, fontsize=10, color=INK2, transform=fig.transFigure)


def show(b: bytes) -> str:
    """Token byte'larını okunur yap: boşluk ␣, satır sonu ↵, yarım UTF-8 karakter �."""
    return b.decode('utf-8', errors='replace').replace(' ', '␣').replace('\n', '↵')


# ---------------------------------------------------------------------------
# BPE eğitimi, her birleştirmeden sonraki dizi uzunluğunu da kaydederek.
# bpe.py / regex_tokenizer.py ile aynı algoritma; regex'te aynı parçalar tek kez sayılıp
# kaç kez geçtiğiyle ağırlıklandırılıyor (sonuç aynı, sadece daha hızlı).
# ---------------------------------------------------------------------------
def train_basic(text, n_merges):
    ids = list(text.encode('utf-8'))
    merges, counts, lengths = {}, [], [len(ids)]
    for i in range(n_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, 256 + i)
        merges[pair] = 256 + i
        counts.append(stats[pair])
        lengths.append(len(ids))
    return merges, counts, lengths


def train_regex(text, pattern, n_merges):
    freq = Counter(re.findall(pattern, text))
    chunks = [list(ch.encode('utf-8')) for ch in freq]
    weights = list(freq.values())
    merges, counts = {}, []
    lengths = [sum(len(c) * w for c, w in zip(chunks, weights))]
    for i in range(n_merges):
        stats = {}
        for c, w in zip(chunks, weights):
            for p in zip(c, c[1:]):
                stats[p] = stats.get(p, 0) + w
        pair = max(stats, key=stats.get)
        chunks = [merge(c, pair, 256 + i) for c in chunks]
        merges[pair] = 256 + i
        counts.append(stats[pair])
        lengths.append(sum(len(c) * w for c, w in zip(chunks, weights)))
    return merges, counts, lengths


def vocab_of(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), i in merges.items():
        vocab[i] = vocab[a] + vocab[b]
    return vocab


# ---------------------------------------------------------------------------
# 1. Görev 1: karakter başına byte
# ---------------------------------------------------------------------------
def fig_utf8():
    text = "Çağrı ılık sütü İzmir'de içti, şöyle güzel 🌸"
    colors = {1: (GRID, INK2), 2: (BLUE_LIGHT, BLUE), 4: (ORANGE_LIGHT, ORANGE)}
    fig, ax = plt.subplots(figsize=(11, 1.9))
    x = 0.0
    for ch in text:
        b = ch.encode('utf-8')
        fc, ec = colors[len(b)]
        w = 0.55 * len(b)
        ax.add_patch(FancyBboxPatch((x, 0), w - 0.08, 1, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc=fc, ec=ec, lw=1))
        label = '␣' if ch == ' ' else ch
        if ch == '🌸':
            label = 'emoji'
        ax.text(x + (w - 0.08) / 2, 0.62, label, ha='center', va='center', fontsize=12 if ch != '🌸' else 8,
                color=INK, fontweight='bold')
        ax.text(x + (w - 0.08) / 2, 0.24, str(len(b)), ha='center', va='center', fontsize=8, color=INK2)
        x += w
    ax.set_xlim(-0.1, x + 0.1)
    ax.set_ylim(-0.15, 1.05)
    ax.axis('off')
    ax.legend(handles=[Patch(fc=colors[k][0], ec=colors[k][1], label=lab) for k, lab in
                       [(1, '1 byte (ASCII)'), (2, '2 byte (ç ğ ı ö ş ü İ)'), (4, '4 byte (emoji)')]],
              loc='upper left', bbox_to_anchor=(0, 0.02), ncol=3, frameon=False, fontsize=9.5)
    n_b = len(text.encode('utf-8'))
    heading(fig, 'Görev 1: Türkçe harfler UTF-8\'de iki byte',
            f'Kutu genişliği = byte sayısı · {len(text)} karakter → {n_b} byte (+%{(n_b / len(text) - 1) * 100:.0f})',
            y=1.24, gap=0.12)
    fig.savefig(f'{OUT}/01_utf8_byte.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Görev 2 + 4: ilk 20 birleştirme, regex'li ve regex'siz
# ---------------------------------------------------------------------------
def merge_kind(tok: bytes):
    s = tok.decode('utf-8', errors='replace')
    if '�' in s or (len(s) == 1 and len(tok) > 1):
        return 'çok byte\'lı karakter kuruluyor'
    if s.endswith(' ') or s.startswith(','):
        return 'kelime sınırını aşıyor'
    if s.startswith(' '):
        return 'boşluk + kelime başı'
    return 'kelime içi'


KIND_COLORS = {
    'çok byte\'lı karakter kuruluyor': ORANGE,
    'kelime sınırını aşıyor': VIOLET,
    'boşluk + kelime başı': AQUA,
    'kelime içi': BLUE,
}


def fig_merges(basic, regex_tr):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharex=False)
    for ax, (name, (merges, counts, _)) in zip(axes, [('BasicTokenizer (regex yok)', basic),
                                                      ('RegexTokenizer (TR deseni)', regex_tr)]):
        vocab = vocab_of(merges)
        items = list(merges.items())[:20]
        ys = list(range(20))[::-1]
        for y, ((a, b), idx), c in zip(ys, items, counts):
            tok = vocab[idx]
            ax.barh(y, c, height=0.72, color=KIND_COLORS[merge_kind(tok)], lw=0)
            ax.text(c + max(counts) * 0.015, y, f'{c:,}', va='center', fontsize=8.5, color=INK2)
        ax.set_yticks(ys, [f'{i + 1:>2}. {show(vocab[idx])}' for i, (_, idx) in enumerate(items)],
                      fontfamily=MONO, fontsize=10)
        ax.tick_params(axis='y', length=0, colors=INK)
        ax.set_xlim(0, max(counts) * 1.15)
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel('çift kaç kez geçiyordu')
        ax.set_title(name, loc='left', fontsize=12, fontweight='bold', color=INK)
    fig.legend(handles=[Patch(color=c, label=k) for k, c in KIND_COLORS.items()], loc='lower center',
               ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.35, bottom=0.12)
    heading(fig, 'Görev 2 ve 4: İlk 20 birleştirme',
            'Tam turkce.txt · ␣ boşluk, � yarım karakter (’ işaretinin ilk 2 byte\'ı) · '
            'regex sınırı aşılamadığı için "e␣", "n␣" yerine "␣b", "␣m" öğreniliyor', y=1.0)
    fig.savefig(f'{OUT}/02_ilk_birlestirmeler.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Görev 2: vocab büyüdükçe sıkıştırma
# ---------------------------------------------------------------------------
def fig_compression(basic, regex_tr, n_bytes):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (_, _, lengths), col, lab, dy in [(basic, BLUE, 'BasicTokenizer', 0.03),
                                          (regex_tr, ORANGE, 'RegexTokenizer (TR)', -0.03)]:
        xs = [256 + i for i in range(len(lengths))]
        ys = [n_bytes / L for L in lengths]
        ax.plot(xs, ys, color=col, lw=2, label=lab)
        ax.plot(xs[-1], ys[-1], 'o', ms=8, color=col, mec=SURFACE, mew=2, zorder=3)
        ax.text(xs[-1] + 4, ys[-1] + dy, f'{ys[-1]:.3f}', va='center', fontsize=10.5, color=INK, fontweight='bold')
    ax.set_xlim(250, 540)
    ax.set_xlabel('vocab boyutu (256 byte + birleştirme)')
    ax.set_ylabel('byte / token')
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left')
    heading(fig, 'Görev 2: Her birleştirme diziyi biraz daha kısaltıyor',
            'Tam turkce.txt · ilk birleştirmeler çok kazandırıyor, sonrakiler giderek daha nadir çiftler · '
            'regex, sınır aşan birleştirmeleri yasakladığı için biraz geride')
    fig.savefig(f'{OUT}/03_sikistirma.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Görev 4-5: aynı metin nasıl bölünüyor
# ---------------------------------------------------------------------------
def token_row(ax, y, pieces, fc, ec, x0=0.0, unit=0.16):
    x = x0
    for p in pieces:
        w = max(len(p), 1) * unit + 0.12
        ax.add_patch(FancyBboxPatch((x, y), w, 0.62, boxstyle='round,pad=0,rounding_size=0.06',
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + 0.31, p, ha='center', va='center', fontsize=9.5, fontfamily=MONO, color=INK)
        x += w + 0.06
    return x


def fig_split(own_vocab, own_tok):
    sent = "Ankara'dan İstanbul'a, Ahmet'te İzmir'deki"
    tr_part = TR.split(',')[0] + ','
    enc = tiktoken.get_encoding('cl100k_base')
    rows = [
        ('GPT-4 regex', 'parçalar', [show(p.encode()) for p in re.findall(GPT4_SPLIT_PATTERN, sent)], ORANGE_LIGHT, ORANGE),
        ('TR regex', 'parçalar', [show(p.encode()) for p in re.findall(TR_SPLIT_PATTERN, sent)], AQUA_LIGHT, AQUA),
        None,
        ('cl100k_base', f'{len(enc.encode(tr_part))} token',
         [show(enc.decode_single_token_bytes(t)) for t in enc.encode(tr_part)], BLUE_LIGHT, BLUE),
        ('kendi (512)', f'{len(own_tok.encode(tr_part))} token',
         [show(own_vocab[t]) for t in own_tok.encode(tr_part)], VIOLET_LIGHT, VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.0))
    y = 0.0
    max_x = 0
    for r in rows:
        if r is None:
            y -= 0.35
            ax.text(0, y + 0.55, f'"{tr_part}"', fontsize=10, color=INK2, fontstyle='italic')
            y -= 0.2
            continue
        name, note, pieces, fc, ec = r
        ax.text(-0.15, y + 0.42, name, ha='right', va='center', fontsize=11, fontweight='bold', color=INK)
        ax.text(-0.15, y + 0.12, note, ha='right', va='center', fontsize=9, color=MUTED)
        max_x = max(max_x, token_row(ax, y, pieces, fc, ec))
        y -= 0.95
    ax.text(0, 0.8, f'"{sent}"', fontsize=10, color=INK2, fontstyle='italic')
    ax.set_xlim(-2.4, max_x + 0.1)
    ax.set_ylim(y + 0.7, 1.05)
    ax.axis('off')
    heading(fig, 'Görev 4–5: Türkçe ekler nasıl bölünüyor',
            'Üstte sadece regex parçaları: İngilizce kısaltma kuralı \'dan → \'d + an yapıyor · '
            'altta GPT-4\'ün tokenizer\'ı ve bizim 512\'lik BPE (TR deseni, turkce.txt)', y=1.04)
    fig.savefig(f'{OUT}/04_bolunme.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Görev 5: token / kelime
# ---------------------------------------------------------------------------
def fig_tr_en(own_tok):
    enc = tiktoken.get_encoding('cl100k_base')
    groups = [('cl100k_base', enc.encode), ('kendi (512)', own_tok.encode)]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    w = 0.36
    for gi, (gname, fn) in enumerate(groups):
        for li, (lang, text, col) in enumerate([('Türkçe', TR, BLUE), ('İngilizce', EN, ORANGE)]):
            n_tok, n_w = len(fn(text)), len(text.split())
            x = gi + (li - 0.5) * (w + 0.03)
            ax.bar(x, n_tok / n_w, width=w, color=col, lw=0, label=lang if gi == 0 else None)
            ax.text(x, n_tok / n_w + 0.08, f'{n_tok / n_w:.2f}', ha='center', fontsize=11, fontweight='bold', color=INK)
            ax.text(x, n_tok / n_w + 0.42, f'{n_tok} token', ha='center', fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1], [g for g, _ in groups], fontsize=11.5, color=INK)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('token / kelime')
    ax.set_ylim(0, 6.3)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left')
    tr_w, en_w = len(TR.split()), len(EN.split())
    ratio = (len(enc.encode(TR)) / tr_w) / (len(enc.encode(EN)) / en_w)
    heading(fig, f'Görev 5: GPT-4 tokenizer\'ında Türkçe kelime başına {ratio:.1f}× pahalı',
            f'Aynı paragraf · TR {tr_w} kelime, EN {en_w} kelime · kendi tokenizer\'ımız Türkçeyi '
            'daha iyi tanıyor ama vocab\'ı ~200× küçük', y=1.08)
    fig.savefig(f'{OUT}/05_tr_en.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Ek (b): GPT eğitim eğrisi, hafta 8 ile aynı birimde
# ---------------------------------------------------------------------------
def fig_gpt():
    with open('samples/turkce_bpe512_video.json', encoding='utf-8') as f:
        r = json.load(f)
    cpt = r['chars_per_token']
    # train_gpt.ipynb eğitim hücresinin çıktısı (her 250 adımda bir) + son değerlendirme
    steps = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    train = [3.3978, 2.6949, 2.2634, 2.0381, 1.8782, 1.7443, 1.6316, 1.5160, r['train']]
    val = [3.5698, 2.9949, 2.6493, 2.5093, 2.4561, 2.4137, 2.4101, 2.4213, r['val']]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(steps, [v / cpt for v in train], color=AQUA, lw=2, marker='o', ms=5, mec=SURFACE, label='train')
    ax.plot(steps, [v / cpt for v in val], color=BLUE, lw=2, marker='o', ms=5, mec=SURFACE, label='val')
    w8 = r['week8_val_per_char']
    ax.axhline(w8, color=ORANGE, lw=1.5, ls='--')
    ax.text(2050, w8 - 0.02, f'hafta 8 val (karakter tokenizer, 5000 adım): {w8:.4f}', ha='right',
            va='top', fontsize=9.5, color=INK2)
    best = min(range(len(val)), key=val.__getitem__)
    ax.plot(steps[best], val[best] / cpt, 'o', ms=10, color=BLUE, mec=SURFACE, mew=2, zorder=3)
    ax.annotate(f'en iyi val: {val[best] / cpt:.4f}\n(step {steps[best]})', xy=(steps[best], val[best] / cpt),
                xytext=(steps[best] + 100, val[best] / cpt + 0.17), fontsize=9.5, color=INK, ha='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    ax.annotate(f'step 0 val = {val[0]:.2f}/token, oysa\nyeni modelde ≈ ln 512 = 6.24\n→ önceki koşudan devam etti',
                xy=(0, val[0] / cpt), xytext=(260, 1.82), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    ax.set_xlabel('adım (bu koşuda)')
    ax.set_ylabel('loss / karakter (nats)')
    ax.set_xlim(-60, 2080)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 1.0))
    heading(fig, 'Ek (b): BPE ile eğitilen GPT, karakter başına hafta 8 ile berabere',
            f'Token başına loss ÷ {cpt:.2f} karakter/token · val step 1500\'den sonra yükseliyor (overfitting) · '
            'Colab L4', y=1.07)
    fig.savefig(f'{OUT}/06_gpt_egitim.png')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    with open(DATA, encoding='utf-8') as f:
        text = f.read()
    n_bytes = len(text.encode('utf-8'))

    basic = train_basic(text, 256)
    regex_tr = train_regex(text, TR_SPLIT_PATTERN, 256)
    own_tok = RegexTokenizer(TR_SPLIT_PATTERN)
    own_tok.merges = regex_tr[0]
    own_tok._build_vocab()

    fig_utf8()
    fig_merges(basic, regex_tr)
    fig_compression(basic, regex_tr, n_bytes)
    fig_split(own_tok.vocab, own_tok)
    fig_tr_en(own_tok)
    fig_gpt()
    print(sorted(os.listdir(OUT)))
