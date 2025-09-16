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

# Environment: set this in Cloud Run
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gcs_bucket02")  # replace in env, not here

# Optional: override which service account signs the URL.
# If omitted, the storage client will attempt to sign with the service account running the service.
SIGNING_SERVICE_ACCOUNT = os.environ.get(
    "SIGNING_SERVICE_ACCOUNT"
)  # e.g. 161693257685-compute@developer.gserviceaccount.com

# Generate filename
def _make_filename():
    return "reply_{}.mp3".format(uuid.uuid4().hex)

# Synthesize text -> mp3 bytes using Google TTS
def synthesize_text_to_mp3(text, voice_name="en-US-Wavenet-D", speaking_rate=1.0):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    # Choose a voice. Change language_code / name as needed.
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name,
        ssml_gender=texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )

    response = client.synthesize_speech(
        request={
            "input": input_text,
            "voice": voice,
            "audio_config": audio_config,
        }
    )

    return response.audio_content  # bytes

# Upload bytes to GCS and return blob name & signed url
def upload_and_get_signed_url(bucket_name, object_name, data_bytes, expiration_minutes=15):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    # Upload bytes
    blob.upload_from_string(data_bytes, content_type="audio/mpeg")

    # Generate signed URL v4. Use service account email if provided (Cloud Run SA),
    # otherwise SDK will try to sign using the environment credentials.
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
        service_account_email=SIGNING_SERVICE_ACCOUNT or None,
    )
    return signed_url

@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    try:
        # ExoTel sends form-urlencoded fields: CallSid, From, text, client_id etc.
        call_sid = request.form.get("CallSid") or request.form.get("callsid")
        from_number = request.form.get("From") or request.form.get("from")
        text = request.form.get("text") or request.form.get("Text") or request.form.get("message")
        client_id = request.form.get("client_id") or request.form.get("ClientId")

        if not text:
            # nothing to say
            return jsonify({"error": "no_text_provided"}), 400

        # Synthesize
        audio_bytes = synthesize_text_to_mp3(text)

        # Choose filename and upload
        filename = _make_filename()
        signed_url = upload_and_get_signed_url(GCS_BUCKET, filename, audio_bytes, expiration_minutes=15)

        # Return JSON that ExoTel expects to play remote URL
        return jsonify({"action": "play_audio_url", "url": signed_url})

    except Exception as e:
        logging.exception("Error in /sip_inbound")
        return jsonify({"error": "internal", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "leadline-webhook", "status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))    response = tts_client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
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
