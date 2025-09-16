# main.py
import os
import uuid
import logging
from datetime import timedelta
from flask import Flask, request, jsonify, send_file
from google.cloud import storage
from google.cloud import texttospeech_v1 as texttospeech
import tempfile

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# -------- CONFIG --------
GCS_BUCKET = os.getenv("GCS_BUCKET", "gcs_bucket02")
SIGNING_SERVICE_ACCOUNT = os.getenv("SIGNING_SERVICE_ACCOUNT")  # optional
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
    """Upload bytes to GCS"""
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

@app.route("/sip_inbound", methods=["POST", "GET"])
def sip_inbound():
    try:
        # Exotel sends form data
        text = (
            request.form.get("text")
            or request.form.get("Text")
            or request.args.get("text")
            or "Hello from AI receptionist"
        )

        logging.info("Received inbound call text=%s", text)

        # Make mp3
        mp3_bytes = synthesize_text_mp3(text)

        # Decide mode
        mode = request.args.get("mode", "json")  # default json, override with ?mode=file

        if mode == "file":
            # Return MP3 directly
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(mp3_bytes)
            tmp.flush()
            return send_file(tmp.name, mimetype="audio/mpeg")

        else:
            # Upload + signed URL
            object_name = f"reply_{uuid.uuid4().hex}.mp3"
            blob = upload_to_gcs(GCS_BUCKET, mp3_bytes, object_name)
            signed_url = make_signed_url_for_blob(blob, SIGNED_URL_MINUTES)

            return jsonify({"action": "play_audio_url", "url": signed_url})

    except Exception as e:
        logging.exception("Error in /sip_inbound")
        return jsonify({"error": "internal", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "leadline-webhook", "status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
