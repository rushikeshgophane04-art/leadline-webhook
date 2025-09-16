# main.py
"""
Single-file Flask app for LeadLine webhook + SIP inbound + simple WebRTC test.
Replace your existing main.py with this file and set env vars in Cloud Run.
Env vars:
  OPENAI_API_KEY
  GOOGLE_SA_JSON        - base64(service-account.json) optional (for Google TTS/STT/Sheets/Storage)
  SHEET_ID              - optional Google Sheets
  DEFAULT_TTS_VOICE     - e.g. "en-IN-Wavenet-C"
  PERSONALAR_ALLOWLIST  - comma-separated numbers (bypass AI and forward to OWNER_NUMBER)
  OWNER_NUMBER          - phone number to forward allowlisted calls to
  PORT                  - optional
"""

import os, json, base64, tempfile, traceback
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
PERSONALAR_ALLOWLIST = os.getenv("PERSONALAR_ALLOWLIST", "")  # comma separated
OWNER_NUMBER    = os.getenv("OWNER_NUMBER", "")              # where to forward allowlist callers
GCS_BUCKET      = os.getenv("GCS_BUCKET")                    # optional: to save audio for training
# ----------------------------

app = Flask(__name__)

# ---------- Init Google clients if provided ----------
g_tts = None
g_stt = None
g_sheet = None
g_storage = None
if GOOGLE_SA_JSON and texttospeech and speech and service_account:
    try:
        sa_json = base64.b64decode(GOOGLE_SA_JSON)
        creds_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(creds_info)
        g_tts = texttospeech.TextToSpeechClient(credentials=creds) if texttospeech else None
        g_stt = speech.SpeechClient(credentials=creds) if speech else None
        if gspread and SHEET_ID:
            gc = gspread.authorize(creds)
            g_sheet = gc.open_by_key(SHEET_ID)
        if storage:
            g_storage = storage.Client(credentials=creds, project=creds_info.get("project_id"))
    except Exception as e:
        app.logger.warning("Google clients init failed: %s" % e)
else:
    if GOOGLE_SA_JSON:
        app.logger.warning("Google libs not installed or missing; GCP features disabled.")

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
    """
    Call OpenAI Chat Completions via REST (requests).
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
    """
    Read small Q/A from Google Sheet tab named client_id or 'default'.
    """
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
        # voice.name expects something like "en-IN-Wavenet-C" — TTS_VOICE env var used
        voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code="en-US")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        app.logger.warning("TTS failed: %s" % e)
        return None

def speech_bytes_to_text(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
    """
    Convert audio bytes to text using Google STT if configured.
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

def save_audio_to_gcs(filename, data_bytes):
    """
    Optional: save audio bytes to GCS bucket for future training.
    """
    if not g_storage or not GCS_BUCKET:
        return None
    try:
        bucket = g_storage.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_string(data_bytes, content_type="audio/mpeg")
        # return public-ish path (not signed) — adjust per your policy
        return f"gs://{GCS_BUCKET}/{filename}"
    except Exception as e:
        app.logger.warning("GCS save failed: %s" % e)
        return None

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def root():
    return jsonify({"service":"leadline-webhook","status":"ok","endpoints":["/webhook (POST)","/sip_inbound (POST)","/webrtc_offer (POST file=audio)"]})

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Generic Dialogflow/Web UI webhook (text-based).
    """
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
    SIP inbound webhook:
    - If caller in PERSONALAR_ALLOWLIST -> forward to OWNER_NUMBER (bypass AI)
    - Else -> STT -> GPT -> TTS -> return base64 MP3 for provider to play
    Response (JSON):
      {"action":"play_audio_base64","audio_b64":"...","text":"..."}
      or
      {"action":"forward","forward_to":"..."}
    """
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        caller = data.get("From") or data.get("caller") or data.get("Caller") or "unknown"
        call_id = data.get("CallSid") or data.get("call_id") or "cid"

        # prepare allowlist
        allowed_numbers = [n.strip() for n in PERSONALAR_ALLOWLIST.split(",") if n.strip()]

        # If caller is explicitly allowlisted and we have an owner number -> forward call
        if caller in allowed_numbers and OWNER_NUMBER:
            return jsonify({
                "call_id": call_id,
                "action": "forward",
                "forward_to": OWNER_NUMBER,
                "text": f"Forwarding call from {caller} to owner."
            })

        # Else handle by AI
        audio_url = data.get("RecordingUrl") or data.get("audio_url")
        user_text = ""
        if audio_url:
            try:
                r = requests.get(audio_url, timeout=20)
                if r.status_code == 200:
                    # note: provider may give mp3/wav; speech_bytes_to_text expects PCM LINEAR16
                    # best-effort: pass raw bytes to STT (Google may or may not accept mp3 here).
                    user_text = speech_bytes_to_text(r.content) or ""
            except Exception as e:
                app.logger.warning("failed download recording: %s" % e)

        if not user_text:
            user_text = data.get("speech_text") or data.get("transcript") or data.get("text") or "Hello"

        client_id = data.get("client_id") or data.get("ClientId")
        answer = call_openai_chat(user_text, client_id=client_id)

        # TTS
        audio_bytes = synthesize_text_to_mp3_bytes(answer)
        audio_b64 = None
        if audio_bytes:
            # optionally save to GCS for training/records
            try:
                # unique-ish filename
                fname = f"call_{call_id[:64]}_{int(__import__('time').time())}.mp3"
                gcs_path = save_audio_to_gcs(fname, audio_bytes) if GCS_BUCKET else None
                if gcs_path:
                    app.logger.info("Saved audio to %s" % gcs_path)
            except Exception:
                app.logger.warning("save to GCS failed")
            audio_b64 = base64.b64encode(audio_bytes).decode()

        return jsonify({
            "call_id": call_id,
            "action": "play_audio_base64",
            "audio_b64": audio_b64,
            "text": answer
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

@app.route("/webrtc_offer", methods=["POST"])
def webrtc_offer():
    """
    Quick test: POST form-data 'audio' (file) to get STT->GPT->TTS MP3 file back.
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error":"POST a file field named 'audio'"}), 400
        f = request.files["audio"]
        audio_bytes = f.read()
        user_text = speech_bytes_to_text(audio_bytes) or "Hello"
        client_id = request.form.get("client_id")
        reply = call_openai_chat(user_text, client_id=client_id)
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
