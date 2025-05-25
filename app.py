from flask import Flask, render_template, request, jsonify
from gtts import gTTS
from dotenv import load_dotenv
import os
import uuid
from openai import OpenAI

from chatbot import ask_bot

app = Flask(__name__)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    reply_text = ask_bot(user_input)

    if not os.path.exists("static"):
        os.makedirs("static")

    tts = gTTS(reply_text)
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = f"static/{audio_filename}"
    tts.save(audio_path)

    # Return path with leading slash for browser
    return jsonify({"reply": reply_text, "audio": f"/static/{audio_filename}"})

@app.route("/voice", methods=["POST"])
def voice():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio file"}), 400

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="text"
    )
    text = transcript.strip()
    reply_text = ask_bot(text)

    if not os.path.exists("static"):
        os.makedirs("static")
    tts = gTTS(reply_text)
    audio_filename = f"{uuid.uuid4()}.mp3"
    audio_path = f"static/{audio_filename}"
    tts.save(audio_path)

    # Return path with leading slash for browser
    return jsonify({"text": text, "reply": reply_text, "audio": f"/static/{audio_filename}"})

if __name__ == "__main__":
    app.run(debug=True)
