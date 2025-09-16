import os
import base64
from flask import Flask, request, jsonify
from google.cloud import texttospeech, storage
from datetime import timedelta

app = Flask(__name__)

# Load environment variables
GCS_BUCKET = os.getenv("GCS_BUCKET", "gcs_bucket02")

# Init Google clients
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"service": "leadline-webhook", "status": "ok"})

@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    try:
        call_sid = request.form.get("CallSid", "test-call")
        text = request.form.get("text", "Hello, this is your AI voice assistant.")

        # Step 1: Generate TTS
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Step 2: Save MP3 to Google Cloud Storage
        filename = f"reply_{call_sid}.mp3"
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_string(response.audio_content, content_type="audio/mpeg")

        # Step 3: Generate signed URL (15 min validity)
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET",
        )

        # Step 4: Return ExoTel instruction
        return jsonify({
            "action": "play_audio_url",
            "url": signed_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
