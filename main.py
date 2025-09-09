# main.py  -- Full-featured LeadLine backend (drop-in)
import os, json, base64, sqlite3, tempfile, time
from flask import Flask, request, jsonify, send_file, g
import openai, requests, logging
from functools import wraps
try:
    from google.cloud import texttospeech, speech_v1p1beta1 as speech
    from google.oauth2 import service_account
    import gspread
except Exception:
    # Google libs optional until GOOGLE_SA_JSON provided
    texttospeech = speech = service_account = gspread = None

# ---------- CONFIG ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY
PORT = int(os.getenv("PORT", 8080))
TEMPLATE_SHEET_ID = os.getenv("TEMPLATE_SHEET_ID")  # template google sheet id (for onboarding)
GOOGLE_SA_JSON = os.getenv("GOOGLE_SA_JSON")  # base64 service account JSON (optional)
DEFAULT_TTS_VOICE = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))  # per-client requests per minute
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")  # admin key to call /onboard_client
# DB path (sqlite for now)
DB_PATH = os.getenv("DB_PATH", "/tmp/leadline.db")
# ----------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- DB helpers ----------
def get_db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS clients(
      id TEXT PRIMARY KEY,
      name TEXT,
      sheet_id TEXT,
      api_token TEXT,
      created_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS usage(
      client_id TEXT,
      period_start INTEGER,
      requests INTEGER,
      tokens_used INTEGER,
      PRIMARY KEY(client_id, period_start)
    );
    """)
    db.commit()

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

# ---------- Google clients (optional) ----------
g_sheets = None
tts_client = None
stt_client = None
if GOOGLE_SA_JSON and texttospeech and speech and service_account:
    try:
        sa = json.loads(base64.b64decode(GOOGLE_SA_JSON))
        creds = service_account.Credentials.from_service_account_info(sa, scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/cloud-platform"
        ])
        tts_client = texttospeech.TextToSpeechClient(credentials=creds)
        stt_client = speech.SpeechClient(credentials=creds)
        gc = gspread.authorize(creds)
        g_sheets = gc
    except Exception as e:
        logging.warning("Google client init failed: %s", e)

# ---------- Utilities ----------
def gen_token():
    import secrets
    return secrets.token_urlsafe(24)

def now_min_period():
    # returns unix minute-aligned integer for usage table
    return int(time.time() // 60 * 60)

def rate_limit_ok(client_id):
    # simple RPM sliding window based on minute buckets
    db = get_db()
    start = now_min_period()
    row = db.execute("SELECT requests FROM usage WHERE client_id = ? AND period_start = ?", (client_id, start)).fetchone()
    if row and row["requests"] >= RATE_LIMIT_RPM:
        return False
    return True

def inc_usage(client_id, tokens=0):
    db = get_db()
    start = now_min_period()
    row = db.execute("SELECT requests,tokens_used FROM usage WHERE client_id=? AND period_start=?", (client_id, start)).fetchone()
    if row:
        db.execute("UPDATE usage SET requests=requests+1, tokens_used=tokens_used+? WHERE client_id=? AND period_start=?", (tokens, client_id, start))
    else:
        db.execute("INSERT INTO usage(client_id,period_start,requests,tokens_used) VALUES(?,?,?,?)", (client_id, start, 1, tokens))
    db.commit()

def fetch_client_row(client_id):
    db = get_db()
    return db.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()

# ---------- Auth decorator ----------
def require_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-API-KEY") or request.args.get("api_key")
        if not token:
            return jsonify({"error":"missing api key"}), 401
        # lookup token -> client_id
        db = get_db()
        row = db.execute("SELECT id FROM clients WHERE api_token = ?", (token,)).fetchone()
        if not row:
            return jsonify({"error":"invalid api key"}), 403
        request.client_id = row["id"]
        return f(*args, **kwargs)
    return wrapper

# ---------- Sheets onboarding ----------
def create_client_sheet(client_id, client_name, client_email=None):
    if not g_sheets:
        return None
    try:
        drive = g_sheets.auth.service_account().open_by_key  # not needed; use API copy via drive API if needed
    except Exception:
        pass
    # fallback: duplicate template via Drive REST
    # using gspread: copy (gspread has limited drive support) -> use Drive REST if required
    # Here we'll use Google Drive copy via requests to Drive API
    sa = json.loads(base64.b64decode(GOOGLE_SA_JSON))
    token_url = "https://oauth2.googleapis.com/token"
    data = {"grant_type":"urn:ietf:params:oauth:grant-type:jwt-bearer","assertion":""}
    # Simpler approach: use Google Drive copy via gspread create new sheet and set headers
    ss = g_sheets.create(f"{client_name} - LeadLine")
    ws = ss.sheet1
    ws.update("A1", [["Question","Answer"]])
    # set sheet permission? skipping for now; you can set drive permissions separately
    return ss.id

# ---------- GPT helper with fallback ----------
def ask_gpt(prompt, client_id=None, max_tokens=300):
    # rate limit check
    cid = client_id or "global"
    if not rate_limit_ok(cid):
        return "Sorry, we are receiving many requests right now. Please try again in a moment."
    # inject client context if available (sheet read)
    context = ""
    if g_sheets and client_id:
        try:
            ss = g_sheets.open_by_key(fetch_client_row(client_id)["sheet_id"])
            try:
                w = ss.worksheet("Sheet1")
            except Exception:
                w = ss.sheet1
            rows = w.get_all_records()
            lines = []
            for r in rows[:30]:
                q = r.get("Question") or r.get("Q") or ""
                a = r.get("Answer") or r.get("A") or ""
                if q and a:
                    lines.append(f"Q: {q}\nA: {a}")
            context = "\n".join(lines)
        except Exception as e:
            logging.info("sheet read failed: %s", e)
    system_msg = {"role":"system","content":"You are a polite professional AI receptionist. Keep replies concise, never use foul language."}
    messages = [system_msg]
    if context:
        messages.append({"role":"system","content":f"Business data:\n{context}"})
    messages.append({"role":"user","content":prompt})
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, max_tokens=max_tokens)
        text = resp['choices'][0]['message']['content'].strip()
        # increment usage (estimate tokens roughly)
        tokens_used = max_tokens if 'usage' not in resp else resp['usage'].get('total_tokens', 0)
        inc_usage(cid, tokens=tokens_used)
        return text
    except Exception as e:
        logging.error("openai error %s", e)
        # fallback canned reply
        return "Sorry, I'm having trouble answering right now. We'll call you back shortly."

# ---------- TTS / STT helpers ----------
def text_to_speech_bytes(text):
    if not tts_client:
        return None
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=DEFAULT_TTS_VOICE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    resp = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return resp.audio_content

def speech_to_text_bytes(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000):
    if not stt_client:
        return ""
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=sample_rate_hz, language_code="en-US")
    resp = stt_client.recognize(config=config, audio=audio)
    texts = []
    for r in resp.results:
        texts.append(r.alternatives[0].transcript)
    return " ".join(texts)

# ---------- Endpoints ----------
@app.route("/")
def root():
    return jsonify({"service":"leadline","status":"ok","endpoints":["/onboard_client","/webhook","/webrtc_offer","/sip_inbound"]})

# Admin: onboard a new client -> create db row + sheet + token
@app.route("/onboard_client", methods=["POST"])
def onboard_client():
    key = request.headers.get("X-ADMIN-KEY")
    if key != ADMIN_API_KEY:
        return jsonify({"error":"unauthorized"}), 403
    j = request.get_json() or {}
    client_id = j.get("id") or j.get("client_id")
    name = j.get("name") or j.get("company") or f"client-{int(time.time())}"
    email = j.get("email")
    token = gen_token()
    sheet_id = None
    if g_sheets:
        sheet_id = create_client_sheet(client_id, name, client_email=email)
    db = get_db()
    db.execute("INSERT OR REPLACE INTO clients(id,name,sheet_id,api_token,created_at) VALUES(?,?,?,?,?)", (client_id, name, sheet_id, token, int(time.time())))
    db.commit()
    return jsonify({"client_id":client_id, "sheet_id":sheet_id, "api_token":token})

# Text webhook (Dialogflow or direct)
@app.route("/webhook", methods=["POST"])
@require_token
def webhook():
    j = request.get_json(force=True, silent=True) or {}
    query = j.get("queryResult",{}).get("queryText","") or j.get("text","") or j.get("message","")
    client_id = request.client_id
    reply = ask_gpt(query, client_id=client_id)
    return jsonify({"fulfillmentText": reply})

# WebRTC demo: accept uploaded audio file, return mp3
@app.route("/webrtc_offer", methods=["POST"])
@require_token
def webrtc_offer():
    client_id = request.client_id
    if "audio" not in request.files:
        return jsonify({"error":"upload form field 'audio' file (wav/pcm)"}), 400
    f = request.files["audio"]
    audio_bytes = f.read()
    # STT
    user_text = speech_to_text_bytes(audio_bytes)
    reply = ask_gpt(user_text, client_id=client_id)
    audio = text_to_speech_bytes(reply)
    if not audio:
        return jsonify({"text":reply})
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio); tmp.flush()
    return send_file(tmp.name, mimetype="audio/mpeg")

# SIP inbound webhook: provider posts RecordingUrl or audio, we return base64 audio to play
@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    # some providers post as form-data, others JSON
    data = request.form.to_dict() or request.json or {}
    token = data.get("api_token") or request.headers.get("X-API-KEY")
    if not token:
        return jsonify({"error":"missing api token"}), 401
    row = get_db().execute("SELECT id FROM clients WHERE api_token=?", (token,)).fetchone()
    if not row:
        return jsonify({"error":"invalid token"}), 403
    client_id = row["id"]
    audio_url = data.get("RecordingUrl") or data.get("audio_url")
    user_text = ""
    if audio_url:
        r = requests.get(audio_url, timeout=15)
        if r.ok:
            user_text = speech_to_text_bytes(r.content)
    else:
        user_text = data.get("speech_text") or data.get("text") or "Hello"
    reply = ask_gpt(user_text, client_id=client_id)
    audio = text_to_speech_bytes(reply)
    b64 = None
    if audio:
        b64 = base64.b64encode(audio).decode()
    return jsonify({"text":reply, "audio_b64": b64})

# Simple health or admin usage query
@app.route("/admin/usage/<client_id>", methods=["GET"])
def admin_usage(client_id):
    key = request.headers.get("X-ADMIN-KEY")
    if key != ADMIN_API_KEY:
        return jsonify({"error":"unauthorized"}), 403
    db = get_db()
    rows = db.execute("SELECT * FROM usage WHERE client_id=? ORDER BY period_start DESC LIMIT 20", (client_id,)).fetchall()
    return jsonify({"usage":[dict(r) for r in rows]})

# Init DB and run
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=PORT)
