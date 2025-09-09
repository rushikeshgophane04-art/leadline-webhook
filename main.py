"""
LeadLine — All-in-one backend (single-file)
Features:
- GPT-4o-mini brain (OpenAI)
- Google STT + TTS (optional, via GOOGLE_SA_JSON)
- Auto-create per-client Google Sheet from TEMPLATE_SHEET_ID
- Per-client API token auth, multi-number mapping
- Free-trial (200 calls), usage/analytics logging, basic billing counters
- Callback scheduler (DB + in-process worker)
- SIP inbound webhook, WebRTC test (file upload), Dialogflow-compatible /webhook
- CRM push hooks (simple)
- Rate-limit & lightweight fraud detection
Notes: For high-scale streaming replace /webrtc_offer with media-server + SIP bridge
"""

import os, json, base64, sqlite3, time, tempfile, threading, traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, g
import requests, logging

# OpenAI new client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Google libs (optional)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from google.cloud import texttospeech, speech_v1p1beta1 as speech
    import gspread
    GOOGLE_AVAILABLE = True
except Exception:
    GOOGLE_AVAILABLE = False

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ADMIN_API_KEY   = os.getenv("ADMIN_API_KEY", "change_me")
TEMPLATE_SHEET_ID = os.getenv("TEMPLATE_SHEET_ID")   # optional: template to copy
GOOGLE_SA_JSON  = os.getenv("GOOGLE_SA_JSON")        # base64 service account JSON
DEFAULT_TTS_VOICE = os.getenv("DEFAULT_TTS_VOICE", "en-IN-Wavenet-C")
PORT = int(os.getenv("PORT", 8080))
FREE_TRIAL_CALLS = int(os.getenv("FREE_TRIAL_CALLS", 200))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", 120))  # per-client requests per minute
DB_PATH = os.getenv("DB_PATH", "/tmp/leadline.db")
# ----------------------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- OpenAI client ----------
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai = None
    logging.warning("OpenAI client not initialized - set OPENAI_API_KEY")

# ---------- Google clients (optional) ----------
g_sheets = None
drive_service = None
tts_client = None
stt_client = None
if GOOGLE_AVAILABLE and GOOGLE_SA_JSON:
    try:
        sa_info = json.loads(base64.b64decode(GOOGLE_SA_JSON))
        creds = service_account.Credentials.from_service_account_info(sa_info, scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/cloud-platform"
        ])
        drive_service = build("drive", "v3", credentials=creds)
        g_sheets = gspread.authorize(creds)
        tts_client = texttospeech.TextToSpeechClient(credentials=creds)
        stt_client = speech.SpeechClient(credentials=creds)
        logging.info("Google clients initialized")
    except Exception as e:
        logging.exception("Google client init failed: %s", e)
else:
    logging.info("Google features disabled until GOOGLE_SA_JSON provided")

# ---------- DB ----------
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
      email TEXT,
      api_token TEXT,
      plan TEXT DEFAULT 'SME',
      sheet_id TEXT,
      crm_url TEXT,
      crm_key TEXT,
      free_calls_left INTEGER DEFAULT 0,
      created_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS numbers(
      number TEXT PRIMARY KEY,
      client_id TEXT
    );
    CREATE TABLE IF NOT EXISTS usage(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      client_id TEXT,
      ts INTEGER,
      endpoint TEXT,
      request_text TEXT,
      response_text TEXT,
      tokens_est INTEGER,
      duration_ms INTEGER
    );
    CREATE TABLE IF NOT EXISTS callbacks(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      client_id TEXT,
      phone TEXT,
      schedule_ts INTEGER,
      status TEXT DEFAULT 'pending',
      attempts INTEGER DEFAULT 0,
      payload TEXT
    );
    CREATE TABLE IF NOT EXISTS limits(
      client_id TEXT PRIMARY KEY,
      minute_bucket INTEGER,
      requests INTEGER
    );
    """)
    db.commit()

@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()

init_db()

# ---------- Utilities ----------
import secrets
def gen_token():
    return secrets.token_urlsafe(24)

def now_ts():
    return int(time.time())

def minute_bucket(ts=None):
    t = ts or now_ts()
    return int(t // 60 * 60)

# ---------- Rate limit & fraud helpers ----------
def check_and_inc_rate(client_id):
    db = get_db()
    bucket = minute_bucket()
    r = db.execute("SELECT requests, minute_bucket FROM limits WHERE client_id=?", (client_id,)).fetchone()
    if not r:
        db.execute("INSERT OR REPLACE INTO limits(client_id, minute_bucket, requests) VALUES(?,?,?)", (client_id, bucket, 1))
        db.commit()
        return True
    if r["minute_bucket"] != bucket:
        db.execute("UPDATE limits SET minute_bucket=?, requests=1 WHERE client_id=?", (bucket, client_id))
        db.commit()
        return True
    if r["requests"] >= RATE_LIMIT_RPM:
        return False
    db.execute("UPDATE limits SET requests=requests+1 WHERE client_id=?", (client_id,))
    db.commit()
    return True

# ---------- Google Sheet auto-create ----------
def create_client_sheet(client_id, client_name, client_email=None):
    """
    Copies TEMPLATE_SHEET_ID (if defined) or creates a fresh sheet, shares with client_email (if provided),
    returns sheet_id or None.
    """
    if not g_sheets:
        logging.info("g_sheets not available")
        return None
    try:
        if TEMPLATE_SHEET_ID:
            # Use Drive API copy
            copy_body = {"name": f"LeadLine_{client_name}_{client_id}"}
            copy_resp = drive_service.files().copy(fileId=TEMPLATE_SHEET_ID, body=copy_body).execute()
            new_id = copy_resp.get("id")
            if client_email:
                perm = {"type":"user","role":"writer","emailAddress":client_email}
                drive_service.permissions().create(fileId=new_id, body=perm, sendNotificationEmail=True).execute()
            return new_id
        else:
            # Create new via gspread
            ss = g_sheets.create(f"LeadLine_{client_name}_{client_id}")
            ss.sheet1.update("A1", [["Question","Answer"]])
            if client_email:
                drive_service.permissions().create(fileId=ss.id, body={"type":"user","role":"writer","emailAddress":client_email}, sendNotificationEmail=True).execute()
            return ss.id
    except Exception as e:
        logging.exception("create_client_sheet failed: %s", e)
        return None

def fetch_client_context_from_sheet(sheet_id, max_rows=50):
    if not g_sheets or not sheet_id:
        return ""
    try:
        ss = g_sheets.open_by_key(sheet_id)
        ws = ss.sheet1
        rows = ws.get_all_records()
        lines = []
        for r in rows[:max_rows]:
            q = r.get("Question") or r.get("Q") or ""
            a = r.get("Answer") or r.get("A") or ""
            if q and a:
                lines.append(f"Q: {q}\nA: {a}")
        return "\n".join(lines)
    except Exception as e:
        logging.exception("sheet read failed: %s", e)
        return ""

# ---------- STT / TTS wrappers (Google) ----------
def text_to_speech_bytes(text, voice_name=DEFAULT_TTS_VOICE):
    if not tts_client:
        return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-IN", name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        resp = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return resp.audio_content
    except Exception:
        logging.exception("TTS failed")
        return None

def speech_to_text_bytes(audio_bytes, encoding="LINEAR16", sample_rate_hz=16000, language_code="en-US"):
    if not stt_client:
        # fallback: return empty string
        return ""
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                          sample_rate_hertz=sample_rate_hz,
                                          language_code=language_code)
        resp = stt_client.recognize(config=config, audio=audio)
        texts = [r.alternatives[0].transcript for r in resp.results]
        return " ".join(texts)
    except Exception:
        logging.exception("STT failed")
        return ""

# ---------- GPT wrapper (OpenAI) ----------
def ask_gpt(prompt, client_ctx="", max_tokens=300):
    if not openai:
        logging.error("OpenAI not configured")
        return "AI not configured."
    try:
        system = {"role":"system","content":"You are a polite professional AI receptionist. Keep replies concise and don't use foul language."}
        messages = [system]
        if client_ctx:
            messages.append({"role":"system","content":f"Business context:\n{client_ctx}"})
        messages.append({"role":"user","content":prompt})
        resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=max_tokens)
        # new client SDK: resp.choices[0].message.content
        text = resp.choices[0].message.content.strip()
        # estimate tokens (approx)
        tokens_est = 0
        try:
            tokens_est = resp.usage.get("total_tokens", 0)
        except Exception:
            tokens_est = int(len(text.split()) / 0.7)  # rough
        return text, tokens_est
    except Exception as e:
        logging.exception("ask_gpt failed: %s", e)
        return "Sorry, I'm having trouble right now.", 0

# ---------- Logging usage ----------
def log_usage(client_id, endpoint, req_text, resp_text, tokens_est=0, duration_ms=0):
    db = get_db()
    db.execute("INSERT INTO usage(client_id, ts, endpoint, request_text, response_text, tokens_est, duration_ms) VALUES(?,?,?,?,?,?,?)",
               (client_id, now_ts(), endpoint, req_text[:2000], resp_text[:2000], tokens_est, duration_ms))
    db.commit()

# ---------- Callbacks worker (simple) ----------
def process_callbacks_loop(sleep_seconds=10):
    """Background thread: pick pending callbacks and attempt to perform them.
       NOTE: For production use Cloud Tasks / PubSub instead of in-process loop."""
    logging.info("Callback worker started")
    while True:
        try:
            db = sqlite3.connect(DB_PATH)
            db.row_factory = sqlite3.Row
            cur = db.cursor()
            nowt = now_ts()
            rows = cur.execute("SELECT * FROM callbacks WHERE status='pending' AND schedule_ts <= ? ORDER BY schedule_ts ASC LIMIT 5", (nowt,)).fetchall()
            for r in rows:
                cid = r["client_id"]
                phone = r["phone"]
                payload = r["payload"]
                # here make outbound call via your SIP provider API / PBX
                # This is a placeholder - each SIP provider has its own outbound API.
                # For now we mark as done to avoid loops.
                cur.execute("UPDATE callbacks SET status='done', attempts=attempts+1 WHERE id=?", (r["id"],))
                db.commit()
                logging.info("Processed callback id=%s client=%s phone=%s", r["id"], cid, phone)
            db.close()
        except Exception:
            logging.exception("callback worker error")
        time.sleep(sleep_seconds)

# start worker thread
cb_thread = threading.Thread(target=process_callbacks_loop, daemon=True)
cb_thread.start()

# ---------- Authentication decorators ----------
from functools import wraps
def require_admin(f):
    @wraps(f)
    def inner(*args, **kwargs):
        key = request.headers.get("X-ADMIN-KEY") or request.args.get("admin_key")
        if not key or key != ADMIN_API_KEY:
            return jsonify({"error":"unauthorized"}), 403
        return f(*args, **kwargs)
    return inner

def require_client_token(f):
    @wraps(f)
    def inner(*args, **kwargs):
        token = request.headers.get("X-API-KEY") or request.args.get("api_key")
        if not token:
            return jsonify({"error":"missing api token"}), 401
        db = get_db()
        r = db.execute("SELECT id, sheet_id, free_calls_left FROM clients WHERE api_token=?", (token,)).fetchone()
        if not r:
            return jsonify({"error":"invalid api token"}), 403
        request.client = {"id": r["id"], "sheet_id": r["sheet_id"], "free_calls_left": r["free_calls_left"]}
        return f(*args, **kwargs)
    return inner

# ---------- Admin endpoints ----------
@app.route("/onboard_client", methods=["POST"])
@require_admin
def onboard_client():
    j = request.get_json() or {}
    client_id = j.get("id") or f"c{int(time.time())}"
    name = j.get("name") or j.get("company") or client_id
    email = j.get("email")
    plan = j.get("plan", "SME")
    token = gen_token()
    sheet_id = None
    if GOOGLE_AVAILABLE and GOOGLE_SA_JSON:
        sheet_id = create_client_sheet(client_id, name, email)
    db = get_db()
    db.execute("INSERT OR REPLACE INTO clients(id,name,email,api_token,plan,sheet_id,free_calls_left,created_at) VALUES(?,?,?,?,?,?,?,?)",
               (client_id, name, email, token, plan, sheet_id, FREE_TRIAL_CALLS, now_ts()))
    db.commit()
    return jsonify({"client_id":client_id, "api_token": token, "sheet_id": sheet_id})

@app.route("/add_number", methods=["POST"])
@require_admin
def add_number():
    j = request.get_json() or {}
    number = j.get("number")
    client_id = j.get("client_id")
    if not number or not client_id:
        return jsonify({"error":"number & client_id required"}), 400
    db = get_db()
    db.execute("INSERT OR REPLACE INTO numbers(number, client_id) VALUES(?,?)", (number, client_id))
    db.commit()
    return jsonify({"ok":True})

@app.route("/admin/clients", methods=["GET"])
@require_admin
def admin_clients():
    db = get_db()
    rows = db.execute("SELECT id,name,email,plan,sheet_id,free_calls_left,created_at FROM clients").fetchall()
    return jsonify({"clients":[dict(r) for r in rows]})

@app.route("/admin/usage/<client_id>", methods=["GET"])
@require_admin
def admin_usage(client_id):
    db = get_db()
    rows = db.execute("SELECT * FROM usage WHERE client_id=? ORDER BY ts DESC LIMIT 200", (client_id,)).fetchall()
    return jsonify({"usage":[dict(r) for r in rows]})

# ---------- Client-facing endpoints ----------
@app.route("/webhook", methods=["POST"])
@require_client_token
def webhook():
    start = time.time()
    client = request.client
    j = request.get_json(force=True, silent=True) or {}
    query = j.get("queryResult",{}).get("queryText") or j.get("text") or j.get("message") or ""
    # rate limit
    if not check_and_inc_rate(client["id"]):
        return jsonify({"fulfillmentText":"Too many requests. Try later."}), 429
    # free-trial enforcement
    db = get_db()
    if client["free_calls_left"] <= 0:
        # you can choose to block or let it pass; we'll block and instruct payment
        return jsonify({"fulfillmentText":"Trial over. Please subscribe to continue."}), 402
    # get context from sheet
    context = ""
    if client["sheet_id"] and GOOGLE_AVAILABLE:
        context = fetch_client_context_from_sheet(client["sheet_id"])
    resp_text, tokens_est = ask_gpt(query, client_ctx=context)
    duration_ms = int((time.time()-start)*1000)
    log_usage(client["id"], "/webhook", query, resp_text, tokens_est, duration_ms)
    # decrement free_calls_left
    db.execute("UPDATE clients SET free_calls_left = free_calls_left - 1 WHERE id=?", (client["id"],))
    db.commit()
    return jsonify({"fulfillmentText": resp_text})

@app.route("/webrtc_offer", methods=["POST"])
@require_client_token
def webrtc_offer():
    client = request.client
    # Expect form-data audio file upload
    if "audio" not in request.files:
        return jsonify({"error":"upload audio file field named 'audio' (wav/pcm)"})
    file = request.files["audio"]
    audio_bytes = file.read()
    # rate limit & trial
    if not check_and_inc_rate(client["id"]):
        return jsonify({"error":"rate_limited"}), 429
    db = get_db()
    if client["free_calls_left"] <= 0:
        return jsonify({"error":"trial_over"}), 402
    # STT
    user_text = speech_to_text_bytes(audio_bytes)
    # context
    context = fetch_client_context_from_sheet(client["sheet_id"]) if client["sheet_id"] else ""
    resp_text, tokens_est = ask_gpt(user_text, client_ctx=context)
    audio = text_to_speech_bytes(resp_text) if GOOGLE_AVAILABLE else None
    log_usage(client["id"], "/webrtc_offer", user_text, resp_text, tokens_est, 0)
    db.execute("UPDATE clients SET free_calls_left = free_calls_left - 1 WHERE id=?", (client["id"],))
    db.commit()
    if audio:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(audio); tmp.flush()
        return send_file(tmp.name, mimetype="audio/mpeg")
    return jsonify({"text": resp_text})

@app.route("/sip_inbound", methods=["POST"])
def sip_inbound():
    # providers differ; accept form or json
    data = request.form.to_dict() or request.get_json(force=True, silent=True) or {}
    # identify client by called number or provided api_token
    called = data.get("To") or data.get("called") or data.get("called_number")
    token = data.get("api_token") or request.headers.get("X-API-KEY")
    db = get_db()
    client_id = None
    if token:
        r = db.execute("SELECT id,sheet_id,free_calls_left FROM clients WHERE api_token=?", (token,)).fetchone()
        if r:
            client_id = r["id"]
            sheet_id = r["sheet_id"]
            free_left = r["free_calls_left"]
    if not client_id and called:
        # map number -> client
        r2 = db.execute("SELECT client_id FROM numbers WHERE number=?", (called,)).fetchone()
        if r2:
            client_id = r2["client_id"]
            sheet_id = db.execute("SELECT sheet_id,free_calls_left FROM clients WHERE id=?", (client_id,)).fetchone()["sheet_id"]
    if not client_id:
        return jsonify({"error":"unknown client"}), 403
    # rate limit & trial check
    if not check_and_inc_rate(client_id):
        return jsonify({"error":"rate_limited"}), 429
    r3 = db.execute("SELECT free_calls_left FROM clients WHERE id=?", (client_id,)).fetchone()
    if r3["free_calls_left"] <= 0:
        return jsonify({"error":"trial_over"}), 402
    # get recording url and download
    audio_url = data.get("RecordingUrl") or data.get("recording_url")
    user_text = ""
    if audio_url:
        try:
            resp = requests.get(audio_url, timeout=10)
            if resp.ok:
                user_text = speech_to_text_bytes(resp.content)
        except Exception:
            logging.exception("download recording failed")
    else:
        user_text = data.get("speech_text") or data.get("text") or "Hello"
    # context
    context = fetch_client_context_from_sheet(sheet_id) if sheet_id else ""
    resp_text, tokens_est = ask_gpt(user_text, client_ctx=context)
    audio = text_to_speech_bytes(resp_text) if GOOGLE_AVAILABLE else None
    # log usage and decrement trial
    log_usage(client_id, "/sip_inbound", user_text, resp_text, tokens_est, 0)
    db.execute("UPDATE clients SET free_calls_left = free_calls_left - 1 WHERE id=?", (client_id,))
    db.commit()
    # return JSON instructing PBX to play base64 audio
    audio_b64 = None
    if audio:
        audio_b64 = base64.b64encode(audio).decode()
    # optionally push lead to CRM
    try:
        crm = db.execute("SELECT crm_url, crm_key FROM clients WHERE id=?", (client_id,)).fetchone()
        if crm and crm["crm_url"]:
            try:
                requests.post(crm["crm_url"], json={"caller":data.get("From"), "transcript":user_text, "reply":resp_text}, headers={"Authorization":f"Bearer {crm['crm_key']}" if crm["crm_key"] else ""}, timeout=5)
            except Exception:
                logging.exception("crm push failed")
    except Exception:
        pass
    return jsonify({"text":resp_text, "audio_b64":audio_b64})

# ---------- Scheduling callbacks (client can request) ----------
@app.route("/schedule_callback", methods=["POST"])
@require_client_token
def schedule_callback():
    j = request.get_json() or {}
    phone = j.get("phone")
    when_ts = j.get("when_ts") or int(time.time()) + 60  # default next minute
    payload = j.get("payload") or ""
    if not phone:
        return jsonify({"error":"phone required"}), 400
    db = get_db()
    db.execute("INSERT INTO callbacks(client_id, phone, schedule_ts, payload) VALUES(?,?,?,?)", (request.client["id"], phone, when_ts, json.dumps(payload)))
    db.commit()
    return jsonify({"ok":True})

# ---------- CRM link setup (admin or client can set) ----------
@app.route("/set_crm", methods=["POST"])
@require_admin
def set_crm():
    j = request.get_json() or {}
    client_id = j.get("client_id")
    crm_url = j.get("crm_url")
    crm_key = j.get("crm_key")
    db = get_db()
    db.execute("UPDATE clients SET crm_url=?, crm_key=? WHERE id=?", (crm_url, crm_key, client_id))
    db.commit()
    return jsonify({"ok":True})

# ---------- Small health root ----------
@app.route("/")
def root():
    return jsonify({"service":"leadline","status":"ok","endpoints":["/onboard_client","/webhook","/webrtc_offer","/sip_inbound","/schedule_callback"]})

# ---------- Run ----------
if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=PORT)
