# main.py
"""
Single-file Flask app for LeadLine webhook + SIP + WebRTC test.
Paste as-is. Set these env vars on Cloud Run:
  OPENAI_API_KEY        - your OpenAI API key (secret)
  GOOGLE_PROJECT        - (optional) GCP project id for Google APIs
  GOOGLE_SA_JSON        - (optional) base64-encoded service account JSON (for Google TTS/STT/Sheets)
  SHEET_ID              - (optional) Google Sheets file ID (one file with tabs per client_id)
  DEFAULT_TTS_VOICE     - e.g. "en-IN-Wavenet-C"
  GCS_BUCKET            - Google Cloud Storage bucket name to upload MP3s
  PORT                  - (optional) container port (Cloud Run provides)
"""
import os, json, base64, tempfile, traceback, uuid, time
from flask import Flask, request, jsonify, send_file
import requests

# Optional Google libs
try:
    from google.cloud import texttospeech, speech_v1p1beta1 as speech, storage as gcs
    from google.oauth2 import service_account
    import gspread
except Exception:
    texttospeech = None
    speech = None
    gcs = None
    service_account = None
    gspread = None

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SA_JSON  = os.getenv("GOOGLE_SA_JSON")    # base64 of service account JSON (optional)
SHEET_ID        = os.getenv("SHEET_ID")          # google sheet id (optional)
TTS_VOICE       = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
GCS_BUCKET      = os.getenv("GCS_BUCKET")        # bucket name to upload mp3s
PORT            = int(os.getenv("PORT", 8080))
# ----------------------------

app = Flask(__name__)

# ---------- Init Google clients if provided ----------
g_tts = None
g_stt = None
g_sheet = None
gcs_client = None
creds = None
if GOOGLE_SA_JSON and service_account:
    try:
        sa_json = base64.b64decode(GOOGLE_SA_JSON)
        creds_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(creds_info)
        # init tts / stt if libs present
        if texttospeech:
            g_tts = texttospeech.TextToSpeechClient(credentials=creds)
        if speech:
            g_stt = speech.SpeechClient(credentials=creds)
        if gspread and SHEET_ID:
            gc = gspread.authorize(creds)
            g_sheet = gc.open_by_key(SHEET_ID)
        # init storage client if lib present + bucket configured
        if gcs and GCS_BUCKET:
            gcs_client = gcs.Client(project=creds_info.get("project_id"), credentials=creds)
    except Exception as e:
        app.logger.warning("Google clients init failed: %s" % e)
else:
    if GOOGLE_SA_JSON:
        app.logger.warning("Google libs not installed or missing dependencies; GCP features disabled.")

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
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
    Use Google TTS -> return mp3 bytes. If GCP not available, return None.
    """
    if not g_tts:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        # language_code left to default; TTS voice contains locale, but set language_code fallback
        voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code="en-US")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        app.logger.warning("TTS failed: %s" % e)
        return None

def speech_bytes_to_text(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
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

def upload_mp3_and_get_signed_url(mp3_bytes, dest_path=None, expires_seconds=3600):
    """
    Upload mp3 bytes to GCS and return a signed URL (v4) valid for expires_seconds.
    Requires gcs_client and creds to be configured and GCS_BUCKET set.
    """
    if not gcs_client or not GCS_BUCKET:
        return None
    try:
        if not dest_path:
            dest_path = f"leadline/{int(time.time())}-{uuid.uuid4().hex}.mp3"
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(dest_path)
        blob.upload_from_string(mp3_bytes, content_type="audio/mpeg")
        # generate signed URL (v4)
        url = blob.generate_signed_url(expiration=expires_seconds, version="v4")
        return url
    except Exception as e:
        app.logger.warning("GCS upload/sign URL failed: %s" % e)
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
        client_id = None
        if "queryResult" in j:
            query_text = j["queryResult"].get("queryText","")
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
    SIP/PBX webhook. Returns JSON that provides ExoTel an audio URL to play.
    Response gives: {"call_id":..., "action":"play_audio_url", "audio_url": "...", "text": "..."}
    If GCS not configured we'll still return base64 (legacy).
    """
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        caller = data.get("From") or data.get("caller") or data.get("Caller") or "unknown"
        call_id = data.get("CallSid") or data.get("call_id") or f"cid-{uuid.uuid4().hex[:8]}"
        audio_url_in = data.get("RecordingUrl") or data.get("audio_url")
        user_text = ""
        if audio_url_in:
            r = requests.get(audio_url_in, timeout=20)
            if r.status_code == 200:
                user_text = speech_bytes_to_text(r.content)
        else:
            user_text = data.get("speech_text") or data.get("transcript") or data.get("text") or "Hello"
        client_id = data.get("client_id") or data.get("ClientId")
        answer = call_openai_chat(user_text, client_id=client_id)
        audio_bytes = synthesize_text_to_mp3_bytes(answer)
        audio_b64 = None
        audio_url = None
        if audio_bytes:
            # try to upload and return signed URL
            audio_url = upload_mp3_and_get_signed_url(audio_bytes)
            if not audio_url:
                # fallback to base64
                audio_b64 = base64.b64encode(audio_bytes).decode()
        # Prefer returning audio_url for ExoTel playback
        if audio_url:
            return jsonify({"call_id":call_id, "action":"play_audio_url", "audio_url": audio_url, "text": answer})
        else:
            return jsonify({"call_id":call_id, "action":"play_audio_base64", "audio_b64": audio_b64, "text": answer})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

@app.route("/webrtc_offer", methods=["POST"])
def webrtc_offer():
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
        # upload to GCS if possible and return URL else return mp3 file
        audio_url = None
        if mp3_bytes:
            audio_url = upload_mp3_and_get_signed_url(mp3_bytes)
        if audio_url:
            return jsonify({"audio_url": audio_url, "text": reply})
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
# ---------- Init Google clients if provided ----------
g_tts = None
g_stt = None
g_sheet = None
if GOOGLE_SA_JSON and texttospeech and speech and service_account and gspread:
    try:
        sa_json = base64.b64decode(GOOGLE_SA_JSON)
        creds_info = json.loads(sa_json)
        creds = service_account.Credentials.from_service_account_info(creds_info)
        g_tts = texttospeech.TextToSpeechClient(credentials=creds)
        g_stt = speech.SpeechClient(credentials=creds)
        if SHEET_ID:
            gc = gspread.authorize(creds)
            g_sheet = gc.open_by_key(SHEET_ID)
        app.logger.info("Google clients initialized.")
    except Exception as e:
        app.logger.warning("Google clients init failed: %s" % e)
else:
    if GOOGLE_SA_JSON:
        app.logger.warning("Google libs not installed or missing dependencies; GCP features disabled.")
    else:
        app.logger.info("No GOOGLE_SA_JSON provided; GCP features disabled.")

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
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
    data = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
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
    """Return mp3 bytes using Google TTS if configured, else None."""
    if not g_tts:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code="en-US")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        app.logger.warning("TTS (mp3) failed: %s" % e)
        return None

def synthesize_text_to_pcm_bytes(text, sample_rate_hz=16000):
    """
    Return raw PCM 16-bit little-endian bytes (LINEAR16) using Google TTS if configured, else None.
    We'll wrap it into a WAV container for SIP playback.
    """
    if not g_tts:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(name=TTS_VOICE, language_code="en-US")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=sample_rate_hz)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content  # raw PCM bytes
    except Exception as e:
        app.logger.warning("TTS (pcm) failed: %s" % e)
        return None

def pcm_bytes_to_wav_bytes(pcm_bytes, sample_rate_hz=16000, channels=1, sampwidth=2):
    """
    Wrap raw PCM bytes into a proper WAV file bytes buffer.
    sampwidth=2 for 16-bit PCM.
    """
    try:
        bio = io.BytesIO()
        with wave.open(bio, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate_hz)
            wf.writeframes(pcm_bytes)
        return bio.getvalue()
    except Exception as e:
        app.logger.warning("pcm->wav failed: %s" % e)
        return None

def speech_bytes_to_text(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
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
    SIP/PBX webhook endpoint. Accept provider form-data or JSON.
    Returns JSON:
      {
        "action":"play_audio_base64",
        "audio_b64_mp3": "...",        # (optional) mp3 base64
        "audio_b64_wav": "...",        # (optional) wav (LINEAR16) base64 - recommended for SIP
        "text": "assistant text"
      }
    ExoTel: configure webhook URL to this endpoint. If ExoTel needs a particular field name, adapt.
    """
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        app.logger.info("sip_inbound payload keys: %s", list(data.keys()))
        caller = data.get("From") or data.get("caller") or data.get("Caller") or "unknown"
        call_id = data.get("CallSid") or data.get("call_id") or "cid"
        audio_url = data.get("RecordingUrl") or data.get("audio_url")
        user_text = ""
        if audio_url:
            # download recorded audio and run STT if possible
            r = requests.get(audio_url, timeout=20)
            if r.status_code == 200:
                user_text = speech_bytes_to_text(r.content)
        else:
            user_text = data.get("speech_text") or data.get("transcript") or data.get("text") or "Hello"
        client_id = data.get("client_id") or data.get("ClientId")
        # Generate AI reply
        answer = call_openai_chat(user_text, client_id=client_id)
        # Try to synthesize both mp3 and wav (WAV wraps LINEAR16 PCM)
        mp3_bytes = synthesize_text_to_mp3_bytes(answer)
        wav_b64 = None
        mp3_b64 = None
        # Prefer PCM->WAV for SIP playback (many SIP gateways expect WAV/PCM)
        pcm = synthesize_text_to_pcm_bytes(answer)
        if pcm:
            wav_bytes = pcm_bytes_to_wav_bytes(pcm)
            if wav_bytes:
                wav_b64 = base64.b64encode(wav_bytes).decode()
        if mp3_bytes:
            mp3_b64 = base64.b64encode(mp3_bytes).decode()
        # Response includes both; provider picks appropriate key
        resp = {
            "call_id": call_id,
            "action": "play_audio_base64",
            "audio_b64_mp3": mp3_b64,
            "audio_b64_wav": wav_b64,
            "text": answer
        }
        app.logger.info("sip_inbound response prepared for call_id=%s mp3=%s wav=%s", call_id, bool(mp3_b64), bool(wav_b64))
        return jsonify(resp)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

@app.route("/webrtc_offer", methods=["POST"])
def webrtc_offer():
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
