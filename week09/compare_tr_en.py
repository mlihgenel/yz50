import tiktoken
from regex_tokenizer import RegexTokenizer, TR_SPLIT_PATTERN

TR = """İstanbul'dan Ankara'ya giden hızlı tren, sabahın erken saatlerinde yola çıktı. \
Yolculardan bazıları pencereden dışarıyı seyrederken, diğerleri kitaplarını okuyordu. \
Kondüktör, trenin gecikmeyeceğini ve öğleden sonra şehre varacaklarını söyledi. \
Çocuklar ise yolculuğun bitmesini sabırsızlıkla bekliyorlardı."""

EN = """The high-speed train from Istanbul to Ankara set off in the early hours of the morning. \
While some of the passengers watched the view through the window, others were reading their books. \
The conductor said that the train would not be delayed and that they would arrive in the city in the afternoon. \
The children, however, were impatiently waiting for the journey to end."""

def statistic(name: str, text: str, encode_fn) -> None:
    n_words = len(text.split())
    n_chars = len(text)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(encode_fn(text))
    print(f"{name:<16} kelime={n_words:>3}  karakter={n_chars:>4}  byte={n_bytes:>4}  "
          f"token={n_tokens:>4}  token/kelime={n_tokens / n_words:.2f}  byte/token={n_bytes / n_tokens:.2f}")

def show_pieces(name: str, ids: list[int], to_bytes, n: int = 30) -> None:
    # her token'ın byte karşılığı; yarım UTF-8 karakter olabileceği için decode'da errors="replace"
    pieces = [to_bytes(idx).decode("utf-8", errors="replace") for idx in ids[:n]]
    print(f"{name}: " + " | ".join(repr(p)[1:-1] for p in pieces))

if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")

    path = "../week08/turkce.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = RegexTokenizer(pattern=TR_SPLIT_PATTERN)
    tok.train(text, 512)

    print("--- token istatistikleri ---")
    statistic("TR / cl100k", TR, enc.encode)
    statistic("EN / cl100k", EN, enc.encode)
    statistic("TR / kendi(512)", TR, tok.encode)
    statistic("EN / kendi(512)", EN, tok.encode)

    print("\n--- TR metnin ilk 30 token'ı ---")
    show_pieces("cl100k", enc.encode(TR), enc.decode_single_token_bytes)
    show_pieces("kendi ", tok.encode(TR), lambda idx: tok.vocab[idx])

    print("\n--- tek kelimeler ---")
    for word in ["İstanbul'dan", "gecikmeyeceğini", "varacaklarını", "sabırsızlıkla", "impatiently"]:
        print(f"{word:<16} cl100k={len(enc.encode(word)):>2}  kendi={len(tok.encode(word)):>2}")
