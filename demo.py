from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

audio_file = open("a.mp3", "rb")
response = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
)

print(response.text)
