# main.py
"""
LeadLine single-file webhook + SIP + WebRTC test.
Env vars (set in Cloud Run):
  OPENAI_API_KEY    - OpenAI API key
  GOOGLE_SA_JSON    - base64(service-account.json) [optional]
  SHEET_ID          - Google Sheet ID (optional)
  DEFAULT_TTS_VOICE - e.g. "en-IN-Wavenet-C" (optional)
  PORT              - container port (default 8080)
  EXOTEL_LANGS      - comma list of langs to prefer Exotel TTS (default "en,hi")
"""

import os, json, base64, tempfile, traceback, time
from flask import Flask, request, jsonify, send_file, Response, url_for
import requests

# Optional Google libs
try:
    from google.cloud import texttospeech, speech_v1p1beta1 as speech
    from google.oauth2 import service_account
    import gspread
except Exception:
    texttospeech = None
    speech = None
    service_account = None
    gspread = None

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_SA_JSON = os.getenv("GOOGLE_SA_JSON")  # base64 JSON
SHEET_ID       = os.getenv("SHEET_ID")
TTS_VOICE      = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
PORT           = int(os.getenv("PORT", 8080))
EXOTEL_LANGS   = os.getenv("EXOTEL_LANGS", "en,hi").split(",")  # languages prefer Exotel TTS
AUDIO_TMP_DIR  = "/tmp/leadline_audio"
os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
# ----------------------------

app = Flask(__name__)

# ---------- Init Google clients ----------
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
        app.logger.info("Google clients initialized")
    except Exception as e:
        app.logger.warning("Google clients init failed: %s", e)
else:
    if GOOGLE_SA_JSON:
        app.logger.warning("Google libs not installed or missing dependencies; GCP features disabled.")

# ---------- Helpers ----------
def call_openai_chat(prompt, client_id=None, model="gpt-4o-mini", max_tokens=300):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    system_msg = {"role": "system", "content": "You are a polite professional AI receptionist. Keep replies concise."}
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
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(resp)[:2000]

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
            a = r.get("Answer") or r.get("A") or r.get("answer") or ""
            if q and a:
                lines.append(f"Q: {q}\nA: {a}")
        return "\n".join(lines)
    except Exception as e:
        app.logger.warning("fetch_client_context error: %s", e)
        return ""

def synthesize_text_to_mp3_bytes(text, voice=TTS_VOICE):
    if not g_tts:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_sel = texttospeech.VoiceSelectionParams(name=voice, language_code="en-US")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = g_tts.synthesize_speech(input=synthesis_input, voice=voice_sel, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        app.logger.warning("TTS failed: %s", e)
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
        resp = g_stt.recognize(config=config, audio=audio)
        texts = [r.alternatives[0].transcript for r in resp.results]
        return " ".join(texts)
    except Exception as e:
        app.logger.warning("STT failed: %s", e)
        return ""

def save_mp3_and_get_public_path(call_id, mp3_bytes):
    fname = f"{call_id}.mp3"
    path = os.path.join(AUDIO_TMP_DIR, fname)
    with open(path, "wb") as fh:
        fh.write(mp3_bytes)
    # public URL using flask url_for (Cloud Run base + route)
    # request.url_root gives base
    base = (request.url_root or "").rstrip("/")
    public = f"{base}/audio/{fname}"
    return public

def make_twiml_play(url):
    tw = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{url}</Play>
</Response>"""
    return Response(tw, content_type="application/xml")

def make_twiml_say(text):
    tw = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{text}</Say>
</Response>"""
    return Response(tw, content_type="application/xml")

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def root():
    return jsonify({"service":"leadline-webhook","status":"ok","endpoints":["/webhook (POST)","/sip_inbound (POST)","/webrtc_offer (POST file=audio)","/audio/<fname>"]})

@app.route("/audio/<fname>", methods=["GET"])
def serve_audio(fname):
    safe = os.path.basename(fname)
    path = os.path.join(AUDIO_TMP_DIR, safe)
    if not os.path.exists(path):
        return jsonify({"error":"not_found"}), 404
    return send_file(path, mimetype="audio/mpeg")

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
    Handles incoming call webhook from Exotel / SIP provider.
    Exotel typically expects TwiML XML in response (<Play> or <Say>).
    Preferred behavior:
     - If request explicitly requests Exotel TTS (prefer_exotel=1) or lang in EXOTEL_LANGS -> return <Say>
     - Else try Google TTS -> save MP3 -> return <Play>public_url
     - If Google TTS fails -> fallback to <Say> text for Exotel TTS
    """
    try:
        data = request.form.to_dict() or request.get_json(silent=True) or {}
        caller = data.get("From") or data.get("caller") or data.get("Caller") or "unknown"
        call_id = data.get("CallSid") or data.get("call_id") or f"call_{int(time.time())}"
        audio_url = data.get("RecordingUrl") or data.get("audio_url")
        lang = (data.get("lang") or data.get("language") or "").lower()
        prefer_exotel = str(data.get("prefer_exotel","")).lower() in ("1","true","yes")
        user_text = ""

        if audio_url:
            r = requests.get(audio_url, timeout=20)
            if r.status_code == 200:
                user_text = speech_bytes_to_text(r.content)
        if not user_text:
            user_text = data.get("speech_text") or data.get("transcript") or data.get("text") or "Hello, how can I help you?"

        client_id = data.get("client_id") or data.get("ClientId")
        answer = call_openai_chat(user_text, client_id=client_id)

        # If prefer Exotel explicitly OR language is in EXOTEL_LANGS -> return Say
        if prefer_exotel or (lang and any(lang.startswith(x.strip().lower()) for x in EXOTEL_LANGS)):
            app.logger.info("Using Exotel TTS (Say) for call %s lang=%s prefer_exotel=%s", call_id, lang, prefer_exotel)
            return make_twiml_say(answer)

        # Try Google TTS
        mp3_bytes = synthesize_text_to_mp3_bytes(answer)
        if mp3_bytes:
            public_url = save_mp3_and_get_public_path(call_id, mp3_bytes)
            app.logger.info("Returning Play URL: %s", public_url)
            return make_twiml_play(public_url)

        # Fallback -> Exotel Say
        app.logger.warning("Google TTS unavailable -> falling back to Exotel Say for call %s", call_id)
        return make_twiml_say(answer)

    except Exception as e:
        app.logger.error("sip_inbound error: %s\n%s", e, traceback.format_exc())
        return make_twiml_say("Sorry, an error occurred. Please try again later.")

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
        tmp.write(mp3_bytes); tmp.flush()
        return send_file(tmp.name, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"internal"}), 500

# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
