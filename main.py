# main.py
"""
Single-file Flask app for LeadLine webhook + SIP + WebRTC test with GCS upload.
Paste as-is. Set these env vars on Cloud Run:
  OPENAI_API_KEY        - your OpenAI API key (secret)
  GOOGLE_SA_JSON        - base64-encoded service account JSON (for Google TTS/STT/Sheets/Storage)
  GCS_BUCKET            - GCS bucket name to upload MP3s (optional)
  SHEET_ID              - (optional) Google Sheets file ID (one file with tabs per client_id)
  DEFAULT_TTS_VOICE     - e.g. "en-IN-Wavenet-C"
  PORT                  - (optional) container port (Cloud Run provides)
  SIGNED_URL_EXPIRATION - signed URL TTL seconds (default 300)
"""

import os, json, base64, tempfile, traceback, time, uuid
from flask import Flask, request, jsonify, send_file
import requests

# Optional Google libs
try:
    from google.cloud import texttospeech, speech_v1p1beta1 as speech
    from google.oauth2 import service_account
    import gspread
    from google.cloud import storage
except Exception:
    texttospeech = None
    speech = None
    service_account = None
    gspread = None
    storage = None

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SA_JSON  = os.getenv("GOOGLE_SA_JSON")    # base64 of service account JSON (optional)
SHEET_ID        = os.getenv("SHEET_ID")          # google sheet id (optional)
TTS_VOICE       = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
PORT            = int(os.getenv("PORT", 8080))
GCS_BUCKET      = os.getenv("GCS_BUCKET")        # e.g. gcs_bucket02
SIGNED_URL_EXPIRATION = int(os.getenv("SIGNED_URL_EXPIRATION", "300"))
# ----------------------------

app = Flask(__name__)

# ---------- Init Google clients if provided ----------
g_tts = None
g_stt = None
g_sheet = None
gcs_client = None
creds_info = None

if GOOGLE_SA_JSON:
    try:
        sa_raw = base64.b64decode(GOOGLE_SA_JSON)
        creds_info = json.loads(sa_raw)
    except Exception as e:
        app.logger.warning("Failed to decode GOOGLE_SA_JSON: %s" % e)
        creds_info = None

if creds_info and service_account:
    try:
        creds = service_account.Credentials.from_service_account_info(creds_info)
        if texttospeech:
            g_tts = texttospeech.TextToSpeechClient(credentials=creds)
        if speech:
            g_stt = speech.SpeechClient(credentials=creds)
        if gspread and SHEET_ID:
            gc = gspread.authorize(creds)
            g_sheet = gc.open_by_key(SHEET_ID)
        if storage:
            # storage client can be created from service account info
            gcs_client = storage.Client.from_service_account_info(creds_info)
    except Exception as e:
        app.logger.warning("Google clients init failed: %s" % e)
else:
    if GOOGLE_SA_JSON:
        app.logger.warning("Google libs not installed or missing dependencies; GCP features disabled.")

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
    """
    Use OpenAI REST ChatCompletions endpoint via requests.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    system_msg = {"role": "system", "content": "You are a polite professional AI receptionist. Be concise."}
    messages = [system_msg]
    if client_id:
        ctx = fetch_client_context(client_id)
        if ctx:
            messages.append({"role":"system","content":f"Business data:\n{ctx}"})
    messages.append({"role":"user","content":prompt})

    url = "https://api.openai.com/v1/chat/completions"
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    resp = r.json()
    try:
        text = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        text = json.dumps(resp)[:2000]
    return text

def fetch_client_context(client_id):
    if not g_sheet:
        return ""
    try:
        try:
            w = g_sheet.worksheet(client_id)
        except Exception:
            w = g_sheet.worksheet("default")
        rows = w.get_all_records()
        lines = []
        for r in rows[:50]:
            q = r.get("Question") or r.get("Q") or r.get("question") or ""
            a = r.get("Answer")   or r.get("A") or r.get("answer") or ""
            if q and a:
                lines.append(f"Q: {q}\nA: {a}")
        return "\n".join(lines)
    except Exception as e:
        app.logger.warning("fetch_client_context error: %s" % e)
        return ""

def synthesize_text_to_mp3_bytes(text):
    """
    Return mp3 bytes using Google TTS if configured, else None.
    """
    if not g_tts:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        # try to infer language_code from voice name prefix if possible
        language_code = "en-US"
        if "-" in TTS_VOICE:
            parts = TTS_VOICE.split("-")
            if len(parts) >= 2:
                # e.g. en-IN-Wavenet-C -> en-IN
                language_code = f"{parts[0]}-{parts[1]}"
        voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code=language_code)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        app.logger.warning("TTS failed: %s" % e)
        return None

def speech_bytes_to_text(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
    """
    Convert raw audio bytes (PCM/WAV) to text using Google STT (if configured).
    """
    if not g_stt:
        return ""
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
            language_code="en-US"
        )
        response = g_stt.recognize(config=config, audio=audio)
        texts = [r.alternatives[0].transcript for r in response.results]
        return " ".join(texts)
    except Exception as e:
        app.logger.warning("STT failed: %s" % e)
        return ""

# ---------- GCS helpers ----------
def upload_bytes_to_gcs(bucket_name, blob_name, data_bytes, content_type="audio/mpeg"):
    """
    Upload bytes to GCS and return uploaded blob name. Requires gcs_client.
    """
    if not gcs_client:
        raise RuntimeError("GCS client not initialized")
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data_bytes, content_type=content_type)
    return blob.name

def generate_signed_url(bucket_name, blob_name, expires_in=SIGNED_URL_EXPIRATION):
    """
    Returns a signed URL valid for expires_in seconds.
    """
    if not gcs_client:
        raise RuntimeError("GCS client not initialized")
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # generate_signed_url exists in google-cloud-storage
    url = blob.generate_signed_url(expiration=expires_in)
    return url

def synthesize_and_upload(text, filename_prefix="reply"):
    """
    Synthesize via Google TTS and upload to GCS; return signed URL or None.
    """
    mp3_bytes = synthesize_text_to_mp3_bytes(text)
    if not mp3_bytes:
        return None
    if not GCS_BUCKET:
        app.logger.warning("GCS_BUCKET not set - returning mp3 bytes only")
        return None
    fname = f"{filename_prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp3"
    try:
        upload_bytes_to_gcs(GCS_BUCKET, fname, mp3_bytes, content_type="audio/mpeg")
        signed = generate_signed_url(GCS_BUCKET, fname)
        return signed
    except Exception as e:
        app.logger.error("upload or signed url failed: %s" % e)
        return None

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def root():
    return jsonify({"service":"leadline-webhook","status":"ok","endpoints":["/webhook (POST)","/sip_inbound (POST)","/webrtc_offer (POST file=audio)"]})

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        j = request.get_json(silent=True) or {}
        query_text = ""
        if "queryResult" in j:
            query_text = j["queryResult"].get("queryText","")
            client_id = None
            session = j.get("session","")
            if session:
                client_id = session.split("/")[-1]
        else:
            query_text = j.get("text") or j.get("query") or j.get("message","")
            client_id = j.get("client_id")
        if not query_text:
            return jsonify({"fulfillmentText":"No query provided."})
        resp_text = call_openai_chat(query_text, client_id=client_id)
        return jsonify({"fulfillmentText": resp_text})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"fulfillmentText": "Error processing request."}), 500

@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    """
    SIP/PBX webhook:
    - Accepts form-data or JSON
    - If provider gives a recording URL, downloads it and STT -> GPT -> TTS -> return signed audio URL
    Response:
      1) {"action":"play_audio_url","audio_url": "...", "text":"..."}  OR
      2) {"action":"play_audio_base64","audio_b64":"...", "text":"..."}
    """
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        caller = data.get("From") or data.get("caller") or data.get("Caller") or "unknown"
        call_id = data.get("CallSid") or data.get("call_id") or "cid"
        audio_url = data.get("RecordingUrl") or data.get("audio_url")
        user_text = ""
        if audio_url:
            try:
                r = requests.get(audio_url, timeout=20)
                if r.status_code == 200:
                    user_text = speech_bytes_to_text(r.content)
            except Exception as e:
                app.logger.warning("download recording failed: %s" % e)
        else:
            user_text = data.get("speech_text") or data.get("transcript") or data.get("text") or "Hello"
        client_id = data.get("client_id") or data.get("ClientId")
        answer = call_openai_chat(user_text, client_id=client_id)
        # Try upload-and-return-signed-url first
        signed_url = None
        try:
            signed_url = synthesize_and_upload(answer, filename_prefix=f"call_{call_id}")
        except Exception as e:
            app.logger.warning("synth+upload failed: %s" % e)
            signed_url = None
        if signed_url:
            return jsonify({"call_id":call_id, "action":"play_audio_url", "audio_url": signed_url, "text": answer})
        else:
            audio_bytes = synthesize_text_to_mp3_bytes(answer)
            audio_b64 = None
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode()
            return jsonify({"call_id":call_id, "action":"play_audio_base64", "audio_b64": audio_b64, "text": answer})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

@app.route("/webrtc_offer", methods=["POST"])
def webrtc_offer():
    """
    Accept an uploaded audio file (form-data 'audio') then:
      - STT -> GPT -> TTS -> returns MP3 file for playback (send_file)
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error":"POST a file field named 'audio'"}), 400
        f = request.files["audio"]
        audio_bytes = f.read()
        user_text = speech_bytes_to_text(audio_bytes) or "Hello"
        client_id = request.form.get("client_id")
        reply = call_openai_chat(user_text, client_id=client_id)
        # attempt to upload to GCS and return signed url
        signed = None
        try:
            signed = synthesize_and_upload(reply, filename_prefix="webrtc")
        except Exception as e:
            app.logger.warning("webrtc synth+upload failed: %s" % e)
            signed = None
        if signed:
            return jsonify({"text": reply, "audio_url": signed})
        mp3_bytes = synthesize_text_to_mp3_bytes(reply)
        if not mp3_bytes:
            return jsonify({"text": reply})
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(mp3_bytes)
        tmp.flush()
        return send_file(tmp.name, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
