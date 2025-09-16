# main.py
from flask import Flask, request, jsonify
from google.cloud import texttospeech, storage
from datetime import timedelta
import os
import logging
import uuid

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

GCS_BUCKET = os.getenv("GCS_BUCKET")  # e.g. gcs_bucket02
# optional allowlist/owner variables (if you use them)
PERSONALAR_ALLOWLIST = os.getenv("PERSONALAR_ALLOWLIST", "")  # comma separated
OWNER_NUMBER = os.getenv("OWNER_NUMBER", "")

# TTS settings (adjust voice/language if desired)
TTS_LANGUAGE_CODE = os.getenv("TTS_LANGUAGE_CODE", "en-US")
TTS_VOICE_NAME = os.getenv("TTS_VOICE_NAME", "en-US-Wavenet-D")  # pick any supported voice
TTS_AUDIO_ENCODING = texttospeech.AudioEncoding.MP3

# short signed url lifetime in minutes
SIGNED_URL_MINUTES = int(os.getenv("SIGNED_URL_MINUTES", "15"))

# clients (use default credentials from Cloud Run)
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()


def synthesize_text_mp3(text: str) -> bytes:
    """Return mp3 bytes from Google Text-to-Speech."""
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE_CODE, name=TTS_VOICE_NAME
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=TTS_AUDIO_ENCODING)
    response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content


def upload_to_gcs(bucket_name: str, content_bytes: bytes, object_name: str) -> None:
    """Upload bytes to GCS (no ACL changes)"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    # write bytes (binary) and set content_type
    blob.upload_from_string(content_bytes, content_type="audio/mpeg")
    return blob


def make_signed_url_for_blob(blob, minutes: int) -> str:
    """Generate a v4 signed URL for GET access (requires signer permissions)."""
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
    )
    return url


@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    try:
        # ExoTel sends form-encoded body (CallSid, From, text, client_id)
        call_sid = request.form.get("CallSid") or request.form.get("callSid") or ""
        from_num = request.form.get("From") or request.form.get("from") or ""
        text = request.form.get("text") or request.form.get("Text") or "Hello"
        client_id = request.form.get("client_id") or request.form.get("clientId") or ""

        logging.info("Incoming call: CallSid=%s From=%s client_id=%s", call_sid, from_num, client_id)

        # (Optional) simple allowlist check for PersonalAR mode
        if PERSONALAR_ALLOWLIST:
            allowed = [p.strip() for p in PERSONALAR_ALLOWLIST.split(",") if p.strip()]
            # if allowed list configured and caller not in list -> fallback behavior (you can adjust)
            if allowed and from_num not in allowed:
                logging.info("Caller not in personal allowlist; proceeding with default flow.")

        # create mp3 bytes with TTS
        mp3_bytes = synthesize_text_mp3(text)

        # object name: use uuid to avoid collision
        object_name = f"reply_{uuid.uuid4().hex}.mp3"
        blob = upload_to_gcs(GCS_BUCKET, mp3_bytes, object_name)

        # generate signed url
        signed_url = make_signed_url_for_blob(blob, SIGNED_URL_MINUTES)

        logging.info("Generated signed URL: %s", signed_url)

        # ExoTel expects JSON like {"action":"play_audio_url","url":"..."}
        return jsonify({"action": "play_audio_url", "url": signed_url})

    except Exception as e:
        logging.exception("Error in sip_inbound")
        return jsonify({"error": "internal", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "leadline-webhook", "status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
