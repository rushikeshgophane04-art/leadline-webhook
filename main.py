# main.py
import os, json, base64, tempfile
from flask import Flask, request, jsonify, send_file
import openai, requests
from google.cloud import texttospeech, speech_v1p1beta1 as speech
from google.oauth2 import service_account
import gspread

# ---------- CONFIG ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY
PORT = int(os.getenv("PORT", 8080))
SHEET_ID = os.getenv("SHEET_ID", None)
GOOGLE_SA_JSON = os.getenv("GOOGLE_SA_JSON", None)  # base64-encoded service account JSON
TTS_VOICE = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
# ----------------------------

app = Flask(__name__)

# init Google clients if creds provided
gcreds = None
if GOOGLE_SA_JSON:
    sa_json = base64.b64decode(GOOGLE_SA_JSON)
    creds = service_account.Credentials.from_service_account_info(json.loads(sa_json))
    tts_client = texttospeech.TextToSpeechClient(credentials=creds)
    stt_client = speech.SpeechClient(credentials=creds)
    if SHEET_ID:
        gs = gspread.authorize(creds)
        sheet = gs.open_by_key(SHEET_ID)
else:
    tts_client = None; stt_client = None; sheet = None

# ----------------------------
# Helper: fetch per-client context (from Google Sheet)
def fetch_client_context(client_id):
    # Example: sheet has tabs named by client_id or a master sheet mapping IDs
    try:
        if not sheet:
            return ""
        try:
            w = sheet.worksheet(client_id)
        except Exception:
            w = sheet.worksheet("default")  # fallback
        rows = w.get_all_records()
        # Build small prompt: Q:.. A:..
        lines = []
        for r in rows[:30]:
            q = r.get("Question") or r.get("Q") or ""
            a = r.get("Answer") or r.get("A") or ""
            if q and a:
                lines.append(f"Q: {q}\nA: {a}")
        return "\n".join(lines)
    except Exception as e:
        return ""

# ----------------------------
# Helper: call OpenAI (gpt-4o-mini)
def ask_gpt(prompt, client_id=None, max_tokens=300):
    system_msg = {
        "role":"system",
        "content":"You are a polite professional AI receptionist. Keep replies concise, never use foul language. If the business context is provided, answer using that context first."
    }
    # inject client context if available
    ctx = fetch_client_context(client_id) if client_id else ""
    messages = [system_msg]
    if ctx:
        messages.append({"role":"system","content":f"Business data:\n{ctx}"})
    messages.append({"role":"user","content":prompt})
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens
    )
    return resp['choices'][0]['message']['content'].strip()

# ----------------------------
# Helper: Text->Speech (Google TTS)
def text_to_speech_bytes(text):
    if not tts_client:
        # fallback: return empty or synthesize via remote TTS
        return None
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=TTS_VOICE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content

# ----------------------------
# Helper: Speech->Text from uploaded audio (Flask file storage)
def speech_to_text_bytes(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
    if not stt_client:
        return ""
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                      sample_rate_hertz=sample_rate_hz,
                                      language_code="en-US")
    resp = stt_client.recognize(config=config, audio=audio)
    texts = []
    for r in resp.results:
        texts.append(r.alternatives[0].transcript)
    return " ".join(texts)

# ----------------------------
# Endpoint: SIP/PBX webhook (POST from your SIP provider)
# Expected fields: from, call_id, audio_url (optional)
@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    data = request.form.to_dict() or request.json or {}
    # basic example fields (adapt to provider)
    caller = data.get("From") or data.get("caller") or "unknown"
    call_id = data.get("CallSid") or data.get("call_id") or "cid"
    # if provider gives recorded audio URL, download and STT it
    audio_url = data.get("RecordingUrl") or data.get("audio_url")
    user_text = ""
    if audio_url:
        r = requests.get(audio_url)
        user_text = speech_to_text_bytes(r.content) if r.status_code==200 else ""
    else:
        # if provider sends DTMF/text or initial greeting, you can use default
        user_text = data.get("speech_text","") or "Hello"

    # call GPT
    client_id = data.get("client_id")  # pass from your PBX routing
    answer = ask_gpt(user_text, client_id=client_id)
    # synthesize
    audio = text_to_speech_bytes(answer)
    # return instruction structure depending on PBX: many PBX accept TwiML-like or JSON
    # We will return a small JSON instructing PBX to play base64 audio
    return jsonify({"call_id":call_id, "action":"play_audio_base64", "audio_b64": base64.b64encode(audio).decode() if audio else None, "text": answer})

# ----------------------------
# Endpoint: WebRTC signalling (very small demo route)
# You will need a client web page that posts SDP offer here and expects an SDP answer
@app.route("/webrtc_offer", methods=["POST"])
def webrtc_offer():
    # In production you use mediasoup/Janus/Pion to handle SDP. Here we accept an uploaded audio file for quick test.
    # Accept file upload ('audio') -> STT -> GPT -> TTS -> return mp3 for playback on client
    if "audio" in request.files:
        f = request.files["audio"]
        user_audio = f.read()
        user_text = speech_to_text_bytes(user_audio)
        client_id = request.form.get("client_id")
        reply = ask_gpt(user_text, client_id=client_id)
        audio = text_to_speech_bytes(reply)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio)
        tmp.flush()
        return send_file(tmp.name, mimetype="audio/mpeg")
    return jsonify({"error":"POST an 'audio' file (wav/pcm)"})

# ----------------------------
# Generic webhook for Dialogflow / UI testing (text-based)
@app.route("/webhook", methods=["POST"])
def webhook():
    j = request.get_json(force=True, silent=True) or {}
    query = j.get("queryResult",{}).get("queryText","") or j.get("text","")
    client_id = j.get("client_id") or j.get("session","").split("/")[-1] if j.get("session") else None
    reply = ask_gpt(query, client_id=client_id)
    return jsonify({"fulfillmentText": reply})

# ----------------------------
@app.route("/")
def root():
    return jsonify({"service":"leadline-webhook","status":"ok","endpoints":["/webhook (POST)","/sip_inbound (POST)","/webrtc_offer (POST file=audio)"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
