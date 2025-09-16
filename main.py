# main.py
"""
LeadLine AI Receptionist
- Handles Exotel/Plivo SIP inbound
- Uses OpenAI for reply
- Uses Google TTS/STT
- Saves audio to GCS bucket (shareable URL)
"""

import os, json, base64, tempfile, traceback, uuid
from flask import Flask, request, jsonify
import requests
from google.cloud import texttospeech, speech_v1p1beta1 as speech
from google.oauth2 import service_account
from google.cloud import storage
import gspread

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SA_JSON  = os.getenv("GOOGLE_SA_JSON")    # base64 service account JSON
SHEET_ID        = os.getenv("SHEET_ID")          # optional
TTS_VOICE       = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
GCS_BUCKET      = os.getenv("GCS_BUCKET")        # required for mp3 upload
OWNER_NUMBER    = os.getenv("OWNER_NUMBER")      # e.g. +919812345678
PERSONAL_ALLOW  = os.getenv("PERSONAL_ALLOW", "")  # comma-separated allowed numbers
PORT            = int(os.getenv("PORT", 8080))
# ----------------------------

app = Flask(__name__)

# ---------- Init Google clients ----------
creds = None
if GOOGLE_SA_JSON:
    sa_json = base64.b64decode(GOOGLE_SA_JSON)
    creds_info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(creds_info)

g_tts = texttospeech.TextToSpeechClient(credentials=creds)
g_stt = speech.SpeechClient(credentials=creds)
gcs_client = storage.Client(credentials=creds)
g_sheet = None
if SHEET_ID:
    gc = gspread.authorize(creds)
    g_sheet = gc.open_by_key(SHEET_ID)

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    system_msg = {"role": "system", "content": "You are a polite professional AI receptionist."}
    messages = [system_msg, {"role": "user", "content": prompt}]
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": messages, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def synthesize_text_to_mp3_bytes(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code="en-US")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

def upload_mp3_to_gcs(audio_bytes, filename=None):
    if not filename:
        filename = f"reply_{uuid.uuid4().hex}.mp3"
    bucket = gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_string(audio_bytes, content_type="audio/mpeg")
    blob.make_public()
    return blob.public_url

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def root():
    return jsonify({"service":"leadline-webhook","status":"ok"})

@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        caller = data.get("From") or "unknown"

        # check allow list
        if OWNER_NUMBER and PERSONAL_ALLOW:
            allow = [n.strip() for n in PERSONAL_ALLOW.split(",")]
            if caller not in allow:
                return jsonify({"call_id": data.get("CallSid","cid"), "action":"hangup"})

        user_text = data.get("speech_text") or data.get("text") or "Hello"
        answer = call_openai_chat(user_text)

        audio_bytes = synthesize_text_to_mp3_bytes(answer)
        audio_url = upload_mp3_to_gcs(audio_bytes)

        return jsonify({
            "call_id": data.get("CallSid","cid"),
            "action": "play_audio_url",
            "audio_url": audio_url,
            "text": answer
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
