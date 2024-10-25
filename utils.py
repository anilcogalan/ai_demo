import asyncio
import openai
import re
from config import OPENAI_API_KEY, MODEL_NAME, SONUC_DOSYASI

# OpenAI API anahtarını ayarla
openai.api_key = OPENAI_API_KEY


def iban_dogrula(iban):
    # Basit bir IBAN doğrulama (Türkiye için)
    iban_pattern = r'^TR\d{2}\d{5}\d{1}\d{16}$'
    return bool(re.match(iban_pattern, iban))


async def bilgi_al():
    while True:
        ad = input("Adınız: ").strip()
        if ad and ad.replace(" ", "").isalpha():
            break
        print("Geçersiz ad. Lütfen sadece harf kullanın.")

    while True:
        soyad = input("Soyadınız: ").strip()
        if soyad and soyad.isalpha():
            break
        print("Geçersiz soyad. Lütfen sadece harf kullanın.")

    while True:
        iban = input("IBAN numaranız: ").strip().upper().replace(" ", "")
        if iban_dogrula(iban):
            break
        print("Geçersiz IBAN. Lütfen geçerli bir Türkiye IBAN'ı girin.")

    return ad, soyad, iban


async def llm_dogrulama(ad, soyad, iban):
    prompt = f"Ad: {ad}\nSoyad: {soyad}\nIBAN: {iban}\n\nLütfen bu bilgileri doğrulayın ve herhangi bir hata veya eksiklik varsa belirtin."

    try:
        response = await openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "Sen bir banka görevlisisin. Müşteri bilgilerini doğrulamakla görevlisin."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Hata oluştu: {str(e)}"


async def sonuc_kaydet(ad, soyad, iban, dogrulama_sonucu):
    with open(SONUC_DOSYASI, "a", encoding="utf-8") as f:
        f.write(f"Ad: {ad}\nSoyad: {soyad}\nIBAN: {iban}\nDoğrulama Sonucu: {dogrulama_sonucu}\n\n")


async def main():
    print("Hoş geldiniz! Lütfen bilgilerinizi girin.")
    ad, soyad, iban = await bilgi_al()

    dogrulama_sonucu = await llm_dogrulama(ad, soyad, iban)
    print("\nDoğrulama Sonucu:")
    print(dogrulama_sonucu)

    await sonuc_kaydet(ad, soyad, iban, dogrulama_sonucu)
    print(f"\nSonuçlar {SONUC_DOSYASI} dosyasına kaydedildi.")


if __name__ == "__main__":
    asyncio.run(main())
