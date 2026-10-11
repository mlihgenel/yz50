# Görev 4: Vikikaynak'tan Türkçe metin (Nutuk, 1927, kamu malı).
# İçindekiler -> bölümler -> kısımlar sırasıyla çekilir, wikitext temizlenir, turkce.txt'ye yazılır.
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://tr.wikisource.org/w/index.php"
ROOT = "Nutuk"
OUT = "turkce.txt"
HEADERS = {'User-Agent': 'yz50-edu/0.1 (educational char-level LM)'}

# python.org kurulumunda sertifika paketi yok; macOS'un kendi paketini kullan
SSL_CTX = ssl.create_default_context()
if os.path.exists('/etc/ssl/cert.pem'):
    SSL_CTX.load_verify_locations('/etc/ssl/cert.pem')


# sayfanın ham wikitext'i. 429 (çok istek) gelirse bekleyip tekrar dener
def fetch_raw(title, retries=6):
    url = BASE + '?' + urllib.parse.urlencode({'title': title, 'action': 'raw'})
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=SSL_CTX) as r:
                return r.read().decode('utf-8')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    yok, atlandı: {title}")
                return ''
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = int(e.headers.get('Retry-After') or 0) or 5 * 2 ** attempt
            print(f"    429, {wait} sn bekleniyor")
            time.sleep(wait)


# içindekilerdeki [[/alt sayfa|...]] linkleri, sırayla ve tam başlık olarak
def subpage_links(wikitext, parent):
    links = re.findall(r'\[\[/([^|\]]+)', wikitext)
    seen = []
    for l in links:
        t = f"{parent}/{l.strip()}"
        if t not in seen:
            seen.append(t)
    return seen


# {{...}} şablonlarını iç içe olanlarla birlikte siler
def strip_templates(s):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith('{{', i):
            depth += 1
            i += 2
        elif s.startswith('}}', i) and depth > 0:
            depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(s[i])
            i += 1
    return ''.join(out)


def clean(wikitext):
    s = strip_templates(wikitext)
    s = re.sub(r'\{\|.*?\|\}', '', s, flags=re.S)                  # tablolar
    s = re.sub(r'<ref[^>]*/>|<ref[^>]*>.*?</ref>', '', s, flags=re.S)  # dipnotlar
    s = re.sub(r'<br\s*/?>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s)                                   # kalan HTML etiketleri
    s = re.sub(r'\[\[(?:Kategori|Category|Dosya|File):[^\]]*\]\]', '', s)
    s = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', s)             # [[hedef|metin]] -> metin
    s = re.sub(r'\[\[([^\]]*)\]\]', r'\1', s)                       # [[metin]] -> metin
    s = re.sub(r"'{2,}", '', s)                                     # ''italik'', '''kalın'''
    s = re.sub(r'^=+\s*(.*?)\s*=+\s*$', r'\1', s, flags=re.M)       # == başlık ==
    s = re.sub(r'^[:*#;]+\s*', '', s, flags=re.M)                   # girinti/liste işaretleri
    s = s.replace('&nbsp;', ' ').replace('﻿', '')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r' *\n *', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


chapters = subpage_links(fetch_raw(ROOT), ROOT)
print(f"{len(chapters)} bölüm")
parts = []
for ch in chapters:
    ch_text = fetch_raw(ch)
    sections = subpage_links(ch_text, ch)
    body = clean(ch_text)   # bölüm sayfasının kendi metni (varsa)
    if body:
        parts.append(body)
    for sec in sections:
        parts.append(clean(fetch_raw(sec)))
        time.sleep(0.5)     # Vikikaynak'a nazik ol
    print(f"{ch}: {len(sections)} kısım, toplam {sum(len(p) for p in parts)} karakter")

text = '\n\n'.join(p for p in parts if p) + '\n'
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"{OUT}: {len(text)} karakter, {len(set(text))} farklı karakter")
