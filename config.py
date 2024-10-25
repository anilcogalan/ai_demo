import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Yapılandırma değişkenlerini tanımla
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"


