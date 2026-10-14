
def analyze_character(char: str) -> dict: 
    raw_bytes = char.encode("utf-8")
    return {
        "char": char, 
        "code_point": ord(char), 
        "byte_count": len(raw_bytes), 
        "bytes_dec" : list(raw_bytes)
    }
    
def print_utf8_table(text: str): 
    print(f"{'Karakter':<10} | {'Unicode Kod':<16} | {'Byte Sayısı':<12} | {'Byte Değerleri (Dec)':<20}")
    print("-" * 65)
    
    one_byte_chars = set()
    two_byte_chars = set()
    
    for char in text:
        info = analyze_character(char)
        display_char = "<boşluk>" if char == " " else char
        unicode_str = f"U+{info['code_point']:04X} ({info['code_point']})"
        bytes_str = str(info["bytes_dec"])
        
        print(f"{display_char:<10} | {unicode_str:<16} | {info['byte_count']:<12} | {bytes_str:<20}")
        
        if info["byte_count"] == 1:
            one_byte_chars.add(char)
        else:
            two_byte_chars.add(char)
            
    print("-" * 65)
    print(f"Toplam Karakter: {len(text)} | Toplam Byte: {len(text.encode('utf-8'))}")
    print(f"1 Byte Tutan Karakterler (ASCII): {sorted(list(one_byte_chars))}")
    print(f"2+ Byte Tutan Karakterler (Özel/Türkçe): {sorted(list(two_byte_chars))}")

if __name__ == "__main__": 
    sample_text = "Hello world, how are you"
    print("--- UTF-8 ve Karakter Analizi Tablosu ---\n")
    print_utf8_table(sample_text)
 