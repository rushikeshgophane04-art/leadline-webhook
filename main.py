# main.py
import os
import uuid
import logging
from datetime import timedelta
from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import texttospeech_v1 as texttospeech

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------- CONFIG --------
GCS_BUCKET = os.getenv("GCS_BUCKET", "gcs_bucket02")
SIGNING_SERVICE_ACCOUNT = os.getenv("SIGNING_SERVICE_ACCOUNT")  # optional
PERSONALAR_ALLOWLIST = os.getenv("PERSONALAR_ALLOWLIST", "")    # e.g. "+911234567890,+919876543210"
OWNER_NUMBER = os.getenv("OWNER_NUMBER", "")                    # optional
SIGNED_URL_MINUTES = int(os.getenv("SIGNED_URL_MINUTES", "15"))
# ------------------------

# Google clients
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()

def synthesize_text_mp3(text: str, voice_name="en-US-Wavenet-D") -> bytes:
    """Convert text -> MP3 bytes using Google TTS"""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
    )
    response = tts_client.synthesize_speech(
        request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
    )
    return response.audio_content

def upload_to_gcs(bucket_name: str, content_bytes: bytes, object_name: str):
    """Upload bytes to GCS (no ACL changes)"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(content_bytes, content_type="audio/mpeg")
    return blob

def make_signed_url_for_blob(blob, minutes: int) -> str:
    """Generate a v4 signed URL for GET access"""
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
        service_account_email=SIGNING_SERVICE_ACCOUNT or None,
    )

@app.route("/sip_inbound", methods=["GET", "POST"])
def sip_inbound():
    try:
        # support both GET & POST (Exotel Passthru uses GET)
        data = request.form if request.method == "POST" else request.args

        call_sid = data.get("CallSid") or ""
        from_num = data.get("From") or ""
        text = data.get("text") or data.get("Text") or "Hello"
        client_id = data.get("client_id") or ""

        logging.info("Incoming call: CallSid=%s From=%s client_id=%s Text=%s",
                     call_sid, from_num, client_id, text)

        # Optional allowlist logic
        if PERSONALAR_ALLOWLIST:
            allowed = [p.strip() for p in PERSONALAR_ALLOWLIST.split(",") if p.strip()]
            if allowed and from_num not in allowed:
                logging.info("Caller not in allowlist; proceeding with default flow.")

        # Create TTS
        mp3_bytes = synthesize_text_mp3(text)

        # Upload and sign URL
        object_name = f"reply_{uuid.uuid4().hex}.mp3"
        blob = upload_to_gcs(GCS_BUCKET, mp3_bytes, object_name)
        signed_url = make_signed_url_for_blob(blob, SIGNED_URL_MINUTES)

        logging.info("Signed URL: %s", signed_url)

        return jsonify({"action": "play_audio_url", "url": signed_url})
    except Exception as e:
        logging.exception("Error in /sip_inbound")
        return jsonify({"error": "internal", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "leadline-webhook", "status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
