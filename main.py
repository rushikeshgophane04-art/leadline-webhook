# main.py - full replacement
import os
import time
import uuid
from datetime import timedelta
from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import texttospeech

# --- Configuration via env vars ---
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gcs_bucket02")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-IN-Wavenet-C")
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "en-IN")
SIGNED_URL_EXPIRE_MIN = int(os.environ.get("SIGNED_URL_EXPIRE_MIN", "15"))

app = Flask(__name__)

# create clients once
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()


def synthesize_text_mp3(text: str) -> bytes:
    """Synthesize text to MP3 bytes using Google Cloud TTS."""
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # choose voice
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANGUAGE,
        name=TTS_VOICE
    )

    # mp3 output
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    return response.audio_content  # bytes


def upload_bytes_to_gcs_and_get_signed_url(bucket_name: str, dest_name: str, data_bytes: bytes, expire_minutes: int):
    """
    Upload bytes to GCS (works with uniform bucket-level access) and return a V4 signed URL.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_name)

    # upload
    blob.upload_from_string(data_bytes, content_type="audio/mpeg")

    # set optional metadata
    blob.metadata = {"uploaded_at": str(int(time.time()))}
    blob.patch()

    # generate signed url (V4)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expire_minutes),
        method="GET"
    )
    return url


@app.route("/", methods=["GET"])
def health():
    return jsonify({"service": "leadline-webhook", "status": "ok"}), 200


@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    """
    ExoTel will POST here.
    Expected form fields (common):
      - CallSid
      - From
      - text  (optional) -- if provided, we will use this text, else fallback message
    Response: JSON with keys action/play_audio_url to let ExoTel play the audio.
    """
    try:
        # read form-data (ExoTel often posts as form data)
        call_sid = request.form.get("CallSid") or request.values.get("CallSid")
        from_number = request.form.get("From") or request.values.get("From")
        text = request.form.get("text") or request.values.get("text") or "Hello, this is an automated assistant. How can I help?"

        # synthesize to mp3 bytes
        mp3_bytes = synthesize_text_mp3(text)

        # destination filename unique per call
        unique_name = f"reply_{uuid.uuid4().hex}.mp3"

        # upload and get signed URL
        signed_url = upload_bytes_to_gcs_and_get_signed_url(GCS_BUCKET, unique_name, mp3_bytes, SIGNED_URL_EXPIRE_MIN)

        # ExoTel-friendly JSON response (adjust keys if your flow expects other keys)
        response_json = {
            "action": "play_audio_url",
            "url": signed_url,
            "filename": unique_name,
            "call_sid": call_sid,
            "from": from_number
        }
        return jsonify(response_json), 200

    except Exception as e:
        app.logger.exception("Error in /sip_inbound")
        return jsonify({"error": "internal", "message": str(e)}), 500


if __name__ == "__main__":
    # local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
