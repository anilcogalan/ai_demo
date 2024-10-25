from datetime import datetime
import streamlit as st
from openai import OpenAI
import re
import os
import sounddevice as sd
import numpy as np
import time
from gtts import gTTS
import base64
from io import BytesIO
from config import OPENAI_API_KEY, MODEL_NAME
import json
import pyaudio
import wave

client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_FOLDER = "audio_records"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Ses kaydı için global değişkenler
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Sabit 5 saniyelik kayıt süresi

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def list_audio_devices():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            # st.sidebar.write(f"Input Device id {i} - {device['name']}")
            pass

def record_audio():
    st.write("Kayıt başlıyor...")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        progress = (i + 1) / (RATE / CHUNK * RECORD_SECONDS)
        st.progress(progress)

    st.write("Kayıt tamamlandı.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

def save_audio(audio_data):
    timestamp = int(time.time())
    filename = f"{AUDIO_FOLDER}/audio_{timestamp}.wav"

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

    st.sidebar.write(f"Ses dosyası kaydedildi: {filename}")
    st.sidebar.write(f"Dosya boyutu: {os.path.getsize(filename)} bytes")

    return filename

def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text",
                language="tr"  # Türkçe dil kodu
            )
        st.sidebar.success(f"Transkripsiyon başarılı: {transcript}")
        return transcript
    except Exception as e:
        st.sidebar.error(f"Transkripsiyon hatası: {e}")
        return None

def message(sender, message_text):
    if sender == "user":
        st.markdown(f'<div class="message user-message"><p>{message_text}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message bot-message"><p>{message_text}</p></div>', unsafe_allow_html=True)

def iban_dogrula(iban):
    iban = iban.replace(' ', '').upper()
    if not re.match(r'^TR\d{24}$', iban):
        return False
    return True

def chat_with_bot(user_input, chat_history):
    messages = [
        {"role": "system", "content": """Sen bir Türkçe konuşan banka asistanısın. Kullanıcıya EFT/havale işlemlerinde yardımcı ol. Her seferinde sadece bir soru sor ve kullanıcının cevabını bekle. Sırasıyla şu bilgileri topla: 1) Alıcının adı, 2) Gönderilecek miktar (TL cinsinden), 3) Alıcının IBAN numarası. Tüm bilgiler tamamlandığında, işlemi özetle ve onay iste. IBAN numarası TR ile başlamalı ve toplam 26 karakter olmalıdır.

        Eğer kullanıcı sadece işlem yapmak istediğini belirtirse (örneğin "Havale yapmak istiyorum"), nazik bir şekilde karşıla ve alıcının adını sor. Kullanıcının verdiği her bilgiyi dikkatle değerlendir ve bir sonraki adıma geç. Eğer bir bilgi eksik veya hatalıysa, nazikçe tekrar sor."""},
    ] + chat_history + [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

def is_valid_name(name):
    return bool(re.match(r'^[A-Za-zÇçĞğİıÖöŞşÜü]+\s+[A-Za-zÇçĞğİıÖöŞşÜü]+(?:\s+[A-Za-zÇçĞğİıÖöŞşÜü]+)*$', name))

def extract_amount(text):
    # Metinden miktar bilgisini çıkar
    match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:TL|Lira|₺)?', text)
    if match:
        amount = match.group(1).replace(',', '.')
        return float(amount)
    return None

def extract_info_with_llm(user_input, current_state):
    prompt = f"""
    Kullanıcının mesajı: "{user_input}"
    
    Mevcut durum:
    - Alıcı: {current_state.get('alici') or 'Henüz belirtilmedi'}
    - Miktar: {current_state.get('miktar') or 'Henüz belirtilmedi'}
    - IBAN: {current_state.get('iban') or 'Henüz belirtilmedi'}
    
    Görevin:
    - Kullanıcının mesajından alıcı adı ve soyadı, miktar ve IBAN bilgilerini çıkar.
    - Alıcı için mutlaka hem ad hem soyad gereklidir. Tek bir isim yeterli değildir.
    - Sonucu aşağıdaki JSON formatında döndür:

    {{
        "alici": Alıcının adı ve soyadı veya null,
        "miktar": Miktar (sayı olarak) veya null,
        "iban": IBAN numarası veya null
    }}

    Not: Sadece istenen JSON formatında çıktı ver, ek açıklama yapma.
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Sen bir banka asistanısın ve sadece EFT/havale işlemlerinde yardımcı olursun."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    try:
        extracted_info = json.loads(response.choices[0].message.content.strip())
        if extracted_info.get('miktar') is not None:
            extracted_info['miktar'] = extract_amount(str(extracted_info['miktar']))
        return extracted_info
    except json.JSONDecodeError:
        return {"alici": None, "miktar": None, "iban": None}

def process_input(user_input):
    if user_input:
        if len(st.session_state.chat_history) == 0:
            initial_response = chat_with_bot(user_input, [])
            bot_response = f"""Merhaba! Ben para gönderme işlemlerinizde size yardımcı olmak için tasarlanmış bir banka asistanıyım. 

Lütfen para göndermek istediğiniz kişinin adını ve soyadını belirtir misiniz?"""
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        elif st.session_state.conversation_state.get('onay'):
            process_confirmation(user_input)
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            if not st.session_state.collected_info.get('alici'):
                if is_valid_name(user_input):
                    st.session_state.collected_info['alici'] = user_input.title()
                    bot_response = f"{user_input.title()} adlı kişiye göndermek istediğiniz miktarı TL cinsinden belirtir misiniz?"
                else:
                    bot_response = "Üzgünüm, girdiğiniz isim geçerli görünmüyor. Lütfen alıcının adını ve soyadını belirtin. Örnek: Ahmet Yılmaz"
            elif not st.session_state.collected_info.get('miktar'):
                amount = extract_amount(user_input)
                if amount is not None:
                    st.session_state.collected_info['miktar'] = amount
                    bot_response = f"{st.session_state.collected_info['alici']} adlı kişiye {amount} TL göndermek istediğinizi anladım. Şimdi, lütfen alıcının IBAN numarasını girer misiniz?"
                else:
                    bot_response = f"Üzgünüm, geçerli bir miktar algılayamadım. Lütfen sadece sayı kullanarak miktarı TL cinsinden belirtin. Örnek: 1000 veya 1000 TL"
            elif not st.session_state.collected_info.get('iban'):
                iban = user_input.replace(" ", "").upper()
                if iban_dogrula(iban):
                    st.session_state.collected_info['iban'] = iban
                    bot_response = confirm_transaction()
                    st.session_state.conversation_state['onay'] = True
                else:
                    bot_response = "Üzgünüm, girdiğiniz IBAN numarası geçerli değil. Lütfen TR ile başlayan 26 haneli geçerli bir IBAN numarası girin."
            else:
                bot_response = "Üzgünüm, anlamadım. Lütfen para göndermek istediğiniz kişinin adını ve soyadını, miktarı veya IBAN numarasını belirtir misiniz?"
            
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.rerun()

def text_to_speech(text):
    tts = gTTS(text=text, lang='tr')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    audio_base64 = base64.b64encode(audio_bytes.read()).decode()
    return audio_base64

def autoplay_audio(audio_base64):
    audio_tag = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_tag, unsafe_allow_html=True)
    time.sleep(0.5)

def confirm_transaction():
    ozet = f"""
    İşlem Özeti:
    - **Alıcı:** {st.session_state.collected_info['alici']}
    - **Miktar:** {st.session_state.collected_info['miktar']} TL
    - **IBAN:** {st.session_state.collected_info['iban']}
    
    Bu bilgiler doğru mu? İşlemi onaylıyor musunuz? (Evet/Hayır)
    """
    return ozet

def process_confirmation(user_input):
    if 'evet' in user_input.lower():
        st.session_state.chat_history.append({"role": "assistant", "content": "İşleminiz başarıyla gerçekleştirilmiştir. Başka bir EFT veya havale işlemi yapmak ister misiniz?"})
        # İşlem geçmişine ekleme
        st.session_state.transaction_history.append({
            'alici': st.session_state.collected_info['alici'],
            'miktar': st.session_state.collected_info['miktar'],
            'iban': st.session_state.collected_info['iban'],
            'tarih': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # Durumu sıfırlama
        st.session_state.collected_info = {'alici': None, 'miktar': None, 'iban': None}
        st.session_state.conversation_state['onay'] = False
    elif 'hayir' in user_input.lower():
        st.session_state.chat_history.append({"role": "assistant", "content": "Anlayışınız için teşekkür ederim. Yardımcı olabileceğim başka bir konu olursa lütfen bana bildirin."})
        # Durumu sıfırlama
        st.session_state.collected_info = {'alici': None, 'miktar': None, 'iban': None}
        st.session_state.conversation_state['onay'] = False
    else:
        st.session_state.chat_history.append({"role": "assistant", "content": "Lütfen 'Evet' veya 'Hayır' şeklinde yanıt verin."})
    st.rerun()

def main():
    st.set_page_config(page_title="EFT/Havale Asistanı", layout="wide")
    local_css("style.css")

    st.title("EFT/Havale Asistanı")

    # Sidebar
    st.sidebar.title("Giriş Yöntemi")
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "Yazı"
    
    new_input_method = st.sidebar.radio("Seçin:", ("Yazı", "Ses"))
    
    if new_input_method != st.session_state.input_method:
        st.session_state.input_method = new_input_method
        st.session_state.user_input = ""
        st.session_state.last_spoken_message_index = -1
        st.rerun()

    # Ses cihazlarını listele
    list_audio_devices()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'collected_info' not in st.session_state:
        st.session_state.collected_info = {'alici': None, 'miktar': None, 'iban': None}
    
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {'onay': False}
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if 'last_spoken_message_index' not in st.session_state:
        st.session_state.last_spoken_message_index = -1

    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = []

    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}

    if 'favorite_recipients' not in st.session_state:
        st.session_state.favorite_recipients = []
    # Açılır kapanır form kısmı
    with st.expander("EFT/Havale Formu", expanded=False):
        alici = st.text_input("Alıcı Adı", value=st.session_state.collected_info.get('alici', ''))
        
        # Miktar değerini float'a çeviriyoruz
        miktar_value = st.session_state.collected_info.get('miktar', 0)
        if isinstance(miktar_value, int):
            miktar_value = float(miktar_value)
        
        miktar = st.number_input("Miktar (TL)", min_value=0.0, step=0.01, value=miktar_value)
        
        iban = st.text_input("IBAN", value=st.session_state.collected_info.get('iban', ''))

        if st.button("İşlemi Gönder", key="send_transaction"):
            if alici and miktar and miktar > 0 and iban and iban_dogrula(iban):
                st.success(f"{alici} adlı kişiye {miktar} TL EFT/havale yapıldı.")
                st.session_state.chat_history.append({"role": "assistant", "content": f"{alici} adlı kişiye {miktar} TL EFT/havale başarıyla yapıldı."})
                # İşlem başarılı olduğunda:
                st.session_state.transaction_history.append({
                    "alici": alici,
                    "miktar": miktar,
                    "iban": iban,
                    "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                # İşlem tamamlandıktan sonra formu temizle
                st.session_state.collected_info = {'alici': None, 'miktar': None, 'iban': None}
            else:
                st.error("Lütfen tüm bilgileri doğru bir şekilde doldurun.")

    # Sohbet
    chat_container = st.container()
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            message(chat["role"], chat["content"])
            if chat["role"] == "assistant" and st.session_state.input_method == "Ses" and i > st.session_state.last_spoken_message_index:
                audio_base64 = text_to_speech(chat["content"])
                autoplay_audio(audio_base64)
                st.session_state.last_spoken_message_index = i

    if st.session_state.input_method == "Yazı":
        user_input = st.text_input("Mesajınızı yazın:", key="text_input", value=st.session_state.user_input)
        if st.button("Mesaj Gönder", key="send_message"):
            if user_input:
                st.session_state.user_input = ""
                process_input(user_input)
    else:
        if st.button("Ses Kaydı Başlat (5 saniye)", key="start_recording"):
            audio_data = record_audio()
            audio_file = save_audio(audio_data)
            st.success("Kayıt tamamlandı ve kaydedildi.")
            
            user_input = transcribe_audio(audio_file)
            if user_input:
                st.write(f"Transkript: {user_input}")
                process_input(user_input)
            else:
                st.error("Ses verisi işlenemedi. Lütfen tekrar deneyin.")

    if st.sidebar.button("Sohbet Geçmişini Temizle", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.collected_info = {
            'alici': None,
            'miktar': None,
            'iban': None
        }
        st.rerun()

    with st.expander("İşlem Geçmişi", expanded=False):
        for islem in st.session_state.transaction_history:
            st.write(f"{islem['tarih']}: {islem['alici']} - {islem['miktar']} TL")

    with st.expander("Sık Kullanılan Alıcılar", expanded=False):
        for alici in st.session_state.favorite_recipients:
            if st.button(f"{alici['ad']} - {alici['iban']}", key=f"fav_{alici['iban']}"):
                st.session_state.conversation_state['alici'] = alici['ad']
                st.session_state.conversation_state['iban'] = alici['iban']
                st.rerun()

if __name__ == "__main__":
    main()
