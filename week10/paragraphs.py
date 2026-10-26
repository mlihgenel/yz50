# hafta 9'daki (compare_tr_en.py) paragraf çifti, aynısı olsun ki sonuçlar bağlansın
TR = """İstanbul'dan Ankara'ya giden hızlı tren, sabahın erken saatlerinde yola çıktı. \
Yolculardan bazıları pencereden dışarıyı seyrederken, diğerleri kitaplarını okuyordu. \
Kondüktör, trenin gecikmeyeceğini ve öğleden sonra şehre varacaklarını söyledi. \
Çocuklar ise yolculuğun bitmesini sabırsızlıkla bekliyorlardı."""

EN = """The high-speed train from Istanbul to Ankara set off in the early hours of the morning. \
While some of the passengers watched the view through the window, others were reading their books. \
The conductor said that the train would not be delayed and that they would arrive in the city in the afternoon. \
The children, however, were impatiently waiting for the journey to end."""

# ek (b) için cümle çiftleri (TR, EN)
PAIRS = [
    ("Bugün hava çok güzel, parkta yürüyüş yapmak istiyorum.",
     "The weather is very nice today, I want to take a walk in the park."),
    ("Türkiye'nin başkenti Ankara'dır ve en kalabalık şehri İstanbul'dur.",
     "The capital of Turkey is Ankara and its most populous city is Istanbul."),
    ("Öğrenciler sınavdan önce bütün gece ders çalıştılar.",
     "The students studied all night before the exam."),
    ("Kitabı okuduktan sonra arkadaşıma ödünç verdim.",
     "After reading the book, I lent it to my friend."),
    ("Yapay zekâ modelleri büyük miktarda metinle eğitiliyor.",
     "Artificial intelligence models are trained on large amounts of text."),
]
