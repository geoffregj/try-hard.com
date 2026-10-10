"""Photogram - LAN media library. Flask + SQLite + Waitress.

Install:  pip install flask waitress            (required)
          pip install pillow                     (optional: photo thumbnails)
          pkg/apt install ffmpeg                 (optional: video thumbnails, MKV/MOV playback)
Run:      python server.py

Environment (all optional):
  HOST=0.0.0.0  PORT=5000  DATA_DIR=<folder next to this file>
  INVITE_CODE=<code needed to create accounts; auto-generated into DATA_DIR/.invite if unset>
  SECRET_KEY=<session key; auto-generated into DATA_DIR/.secret if unset>
  MAX_FILE_GB=50  SECURE_COOKIES=1 (set only when served over HTTPS)
  IMPORT_DIR=<folder the admin drops files into, default DATA_DIR/import>
  IMPORT_MODE=move|link   (move = take the files into the library; link = leave them where they are)
  DEFAULT_QUOTA_GB=0   (per-user storage cap for non-admins, 0 = unlimited; admin can override per user)
The first account created (or the oldest existing one) is the admin.
"""
import hashlib, hmac, logging, os, re, secrets, shutil, socket, sqlite3, subprocess, threading, time, uuid
from flask import (Flask, Response, abort, g, jsonify, redirect, request,
                   send_from_directory, session)
from markupsafe import escape

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.environ.get("DATA_DIR") or BASE)
UPLOADS, TMP = os.path.join(DATA, "uploads"), os.path.join(DATA, "tmp_uploads")
THUMBS, CACHE = os.path.join(DATA, "thumbs"), os.path.join(DATA, "cache")
DB = os.path.join(DATA, "photogram.db")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
MAX_FILE = int(float(os.environ.get("MAX_FILE_GB", "50")) * 1024**3)
DEFAULT_QUOTA_GB = float(os.environ.get("DEFAULT_QUOTA_GB", "0"))
IMPORT_DIR = os.path.abspath(os.environ.get("IMPORT_DIR") or os.path.join(DATA, "import"))
IMPORT_MODE = "link" if os.environ.get("IMPORT_MODE", "move").lower() == "link" else "move"
REQ_LIMIT = 40 * 1024 * 1024              # biggest single request (one chunk)
MAX_SESSIONS_PER_USER = 50
FFMPEG = shutil.which("ffmpeg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("photogram")

for d in (DATA, UPLOADS, THUMBS, CACHE, IMPORT_DIR):
    os.makedirs(d, exist_ok=True)
shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)   # partial uploads die with the process

UP = {}                                   # active chunked uploads (in memory)
JOBS = {}                                 # MKV/MOV conversions: media id -> "working" | "failed"
JLOCK = threading.Lock()
CONVERT_SLOT = threading.Semaphore(1)     # one ffmpeg conversion at a time
NOTHUMB = set()
IMPORT = {"state": "idle", "done": 0, "skipped": 0, "msg": ""}; ILOCK = threading.Lock()

KIND = {**dict.fromkeys(["mp4", "mov", "webm", "mkv"], "video"),
        **dict.fromkeys(["mp3", "m4a", "wav", "ogg", "flac"], "audio"),
        **dict.fromkeys(["jpg", "png", "gif", "webp"], "image")}

def secret_file(name, make):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        with open(p, "w") as f: f.write(make())
    try: os.chmod(p, 0o600)
    except OSError: pass
    with open(p) as f: return f.read().strip()

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=REQ_LIMIT, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("SECURE_COOKIES") == "1",
                  PERMANENT_SESSION_LIFETIME=30 * 24 * 3600)
app.secret_key = os.environ.get("SECRET_KEY") or secret_file(".secret", lambda: secrets.token_hex(32))
INVITE = os.environ.get("INVITE_CODE") or secret_file(".invite", lambda: secrets.token_urlsafe(6))

# ---------- db ----------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB, timeout=10); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d: d.close()

def init_db():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL COLLATE NOCASE,
      salt TEXT NOT NULL, pin_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS media(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
      stored TEXT NOT NULL UNIQUE, orig TEXT NOT NULL, size INTEGER NOT NULL,
      kind TEXT NOT NULL, ext TEXT NOT NULL, created REAL NOT NULL, series TEXT);""")
    if "series" not in [r[1] for r in c.execute("PRAGMA table_info(media)")]:
        c.execute("ALTER TABLE media ADD COLUMN series TEXT")      # upgrade an older database in place
    ucols = [r[1] for r in c.execute("PRAGMA table_info(users)")]
    if "is_admin" not in ucols: c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
    if "quota_gb" not in ucols: c.execute("ALTER TABLE users ADD COLUMN quota_gb REAL")
    c.execute("CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts REAL NOT NULL, user_id INTEGER, "
              "user TEXT, ip TEXT, action TEXT NOT NULL, detail TEXT)")
    if not c.execute("SELECT 1 FROM users WHERE is_admin=1").fetchone():
        c.execute("UPDATE users SET is_admin=1 WHERE id=(SELECT MIN(id) FROM users)")
    c.executescript("CREATE INDEX IF NOT EXISTS i_media_created ON media(created);"
                    "CREATE INDEX IF NOT EXISTS i_media_user_series ON media(user_id, series);")
    c.commit(); c.close()

# ---------- security ----------
def hash_pin(pin, salt):
    return "s1$" + hashlib.scrypt(pin.encode(), salt=salt.encode(), n=2**14, r=8, p=1, dklen=32).hex()

def check_pin(u, pin):
    if u["pin_hash"].startswith("s1$"):
        return hmac.compare_digest(u["pin_hash"], hash_pin(pin, u["salt"]))
    old = hashlib.sha256((u["salt"] + pin).encode()).hexdigest()     # legacy hash: verify, then upgrade
    if hmac.compare_digest(u["pin_hash"], old):
        db().execute("UPDATE users SET pin_hash=? WHERE id=?", (hash_pin(pin, u["salt"]), u["id"])); db().commit()
        return True
    return False

def sniff(h):
    """File type from its first bytes, never from the filename."""
    if h.startswith(b"\xff\xd8\xff"): return "jpg"
    if h.startswith(b"\x89PNG\r\n\x1a\n"): return "png"
    if h[:6] in (b"GIF87a", b"GIF89a"): return "gif"
    if h[:4] == b"RIFF":
        if h[8:12] == b"WEBP": return "webp"
        if h[8:12] == b"WAVE": return "wav"
    if h[4:8] == b"ftyp":
        b = h[8:12]
        if b in (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif", b"avis"):
            return None                                              # HEIF/AVIF stills are not MP4 video
        return "mov" if b == b"qt  " else "m4a" if b in (b"M4A ", b"M4B ") else "mp4"
    if h[:4] == b"\x1a\x45\xdf\xa3": return "webm" if b"webm" in h else "mkv"
    if h[:3] == b"ID3" or (len(h) > 1 and h[0] == 0xff and h[1] & 0xe0 == 0xe0): return "mp3"
    if h[:4] == b"OggS": return "ogg"
    if h[:4] == b"fLaC": return "flac"
    return None

_hits, _hlock = {}, threading.Lock()
def rate_limited(key, limit=8, window=60):
    now = time.time()
    with _hlock:
        if len(_hits) > 5000:                                        # keep memory bounded
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > window]: _hits.pop(k, None)
        _hits[key] = [t for t in _hits.get(key, []) if now - t < window]
        if len(_hits[key]) >= limit: return True
        _hits[key].append(now); return False

def csrf_token():
    if "csrf" not in session: session["csrf"] = secrets.token_hex(16)
    return session["csrf"]

@app.before_request
def guard():
    if request.method == "POST":
        t = request.headers.get("X-CSRF") or request.form.get("csrf", "")
        if not hmac.compare_digest(t.encode(), session.get("csrf", "x").encode()): abort(403)

@app.after_request
def headers(r):
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "same-origin"
    r.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; media-src 'self'; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
    if request.path.startswith("/api/"): r.headers["Cache-Control"] = "no-store"
    return r

def wants_json(): return request.path.startswith(("/api/", "/media/", "/thumb/", "/cache/"))

@app.errorhandler(401)
def e401(_): return (jsonify(error="Sign in required."), 401) if wants_json() else redirect("/login")
@app.errorhandler(403)
def e403(_): return jsonify(error="Forbidden."), 403
@app.errorhandler(404)
def e404(_): return jsonify(error="Not found."), 404
@app.errorhandler(413)
def e413(_): return jsonify(error="Request too large."), 413
@app.errorhandler(500)
def e500(e):
    log.exception("Unhandled error: %s", e); return jsonify(error="Server error."), 500

@app.route("/healthz")
def healthz(): return "ok"

@app.route("/favicon.ico")
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#fdbd27"/>'
           '<path d="M12 9l12 7-12 7z" fill="#102343"/></svg>')
    return Response(svg, mimetype="image/svg+xml", headers={"Cache-Control": "max-age=86400"})

def client_ip():
    """Real visitor IP. Only trust Cloudflare's header when the request came from
    this machine (cloudflared); LAN clients can't spoof it."""
    ra = request.remote_addr
    if ra in ("127.0.0.1", "::1"):
        return request.headers.get("CF-Connecting-IP") or ra
    return ra

def me(): return session.get("uid")
def need_login():
    if not me(): abort(401)
    u = db().execute("SELECT id, name, is_admin, quota_gb FROM users WHERE id=?", (me(),)).fetchone()
    if not u: session.clear(); abort(401)          # account was deleted
    g.user = u
def is_admin(): return bool(g.user["is_admin"])
def need_admin():
    need_login()
    if not is_admin(): abort(403)
def quota_bytes(u):
    if u["is_admin"]: return 0
    q = u["quota_gb"] if u["quota_gb"] is not None else DEFAULT_QUOTA_GB
    return int(q * 1024**3)                          # 0 = unlimited

def audit(action, detail="", uid=None, name=None):
    try:
        d = db()
        d.execute("INSERT INTO audit(ts,user_id,user,ip,action,detail) VALUES(?,?,?,?,?,?)",
                  (time.time(), uid or me(), name or session.get("name"), client_ip(), action, str(detail)[:300]))
        d.execute("DELETE FROM audit WHERE id <= (SELECT MAX(id) FROM audit) - 5000")
        d.commit()
    except Exception: log.exception("audit failed")

# ---------- maintenance ----------
def sweep():
    """Drop uploads abandoned for over 6 hours."""
    now = time.time()
    for k, s in list(UP.items()):
        try: idle = now - os.path.getmtime(s["path"])
        except OSError: idle = 1e9
        if idle > 6 * 3600:
            UP.pop(k, None)
            try: os.remove(s["path"])
            except OSError: pass

def sweeper():
    while True:
        time.sleep(600)
        try: sweep()
        except Exception: log.exception("sweep failed")

def purge(stored):
    stem = stored.rsplit(".", 1)[0]
    for p in (os.path.join(UPLOADS, stored), os.path.join(THUMBS, stem + ".jpg"), os.path.join(CACHE, stem + ".mp4")):
        try: os.remove(p)
        except OSError: pass
    NOTHUMB.discard(stem)

def make_thumb(name, kind):
    stem = name.rsplit(".", 1)[0]; src = os.path.join(UPLOADS, name); dst = os.path.join(THUMBS, stem + ".jpg")
    try:
        if kind == "image":
            from PIL import Image
            with Image.open(src) as im:
                im.thumbnail((240, 240)); im.convert("RGB").save(dst, "JPEG", quality=75)
        elif kind == "video" and FFMPEG:
            for ss in ("3", "0"):
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", ss, "-i", src, "-frames:v", "1",
                                "-vf", "scale=240:-2", dst], timeout=60, capture_output=True)
                if os.path.exists(dst): break
    except Exception: pass
    if not os.path.exists(dst): NOTHUMB.add(stem)

# ---------- pages ----------
@app.route("/")
def index():
    need_login()
    nav = "<a data-v=adm><i>&#9881;</i>Admin</a>" if is_admin() else ""
    return (PAGE.replace("__ADMINNAV__", nav).replace("__CSRF__", csrf_token())
                .replace("__NAME__", str(escape(session["name"]))).replace("__INIT__", str(escape(session["name"][:1].upper()))))

@app.route("/login", methods=["GET", "POST"])
def login():
    err = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()[:24]
        pin = request.form.get("pin", "")
        if rate_limited("ip:" + str(client_ip())) or rate_limited("n:" + name.lower(), limit=10):
            err = "Too many attempts. Wait a minute."
        elif not name or not (pin.isdigit() and 4 <= len(pin) <= 8):
            err = "Name required. PIN must be 4-8 digits."
        else:
            u = db().execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
            if u is None:
                if not hmac.compare_digest(request.form.get("invite", "").strip().encode(), INVITE.encode()):
                    err = "New accounts need a valid invite code."; audit("signup_denied", name, name=name)
                else:
                    salt = secrets.token_hex(8)
                    try:
                        first = db().execute("SELECT 1 FROM users WHERE is_admin=1").fetchone() is None
                        cur = db().execute("INSERT INTO users(name,salt,pin_hash,is_admin) VALUES(?,?,?,?)",
                                           (name, salt, hash_pin(pin, salt), int(first)))
                        db().commit()
                    except sqlite3.IntegrityError:
                        err = "That name was just taken. Try again."
                    else:
                        session.clear(); session.permanent = True
                        session["uid"] = cur.lastrowid; session["name"] = name
                        log.info("new user %s", name); audit("signup", uid=cur.lastrowid, name=name); return redirect("/")
            elif check_pin(u, pin):
                session.clear(); session.permanent = True
                session["uid"] = u["id"]; session["name"] = u["name"]
                audit("login"); return redirect("/")
            else:
                err = "Wrong PIN."; log.warning("failed login for %s from %s", name, client_ip()); audit("login_failed", uid=u["id"], name=u["name"])
    box = f'<div class=err>{escape(err)}</div>' if err else ""
    return LOGIN.replace("__ERR__", box).replace("__CSRF__", csrf_token())

@app.route("/logout", methods=["POST"])
def logout():
    session.clear(); return redirect("/login")

# ---------- library ----------
@app.route("/api/media")
def media():
    need_login()
    rows = db().execute("""SELECT m.id, m.user_id, m.orig, m.size, m.kind, m.ext, m.created, m.stored, m.series,
        u.name AS owner, m.user_id=? AS mine FROM media m JOIN users u ON u.id=m.user_id
        ORDER BY m.created DESC""", (me(),)).fetchall()
    return jsonify([{**dict(r), "mine": bool(r["mine"]) or is_admin()} for r in rows])

@app.route("/api/stats")
def stats():
    need_login()
    used = db().execute("SELECT COALESCE(SUM(size),0) FROM media").fetchone()[0]
    mine = db().execute("SELECT COALESCE(SUM(size),0) FROM media WHERE user_id=?", (me(),)).fetchone()[0]
    return jsonify(used=used, free=shutil.disk_usage(DATA).free, ffmpeg=bool(FFMPEG), mine=mine, quota=quota_bytes(g.user))

@app.route("/media/<name>")
def get_media(name):
    need_login()
    r = db().execute("SELECT orig FROM media WHERE stored=?", (name,)).fetchone()
    if not r: abort(404)
    dl = request.args.get("dl") == "1"   # Range requests supported -> seeking works
    return send_from_directory(UPLOADS, name, as_attachment=dl, download_name=r["orig"])

@app.route("/thumb/<name>")
def thumb(name):
    need_login()
    stem = name.rsplit(".", 1)[0]
    if not re.fullmatch(r"[0-9a-f]{32}", stem) or stem in NOTHUMB: abort(404)
    if not os.path.exists(os.path.join(THUMBS, stem + ".jpg")):
        r = db().execute("SELECT stored, kind FROM media WHERE stored LIKE ?", (stem + ".%",)).fetchone()
        if not r or r["kind"] == "audio": abort(404)
        make_thumb(r["stored"], r["kind"])                           # backfills older uploads on demand
    return send_from_directory(THUMBS, stem + ".jpg", max_age=86400)

@app.route("/cache/<name>")
def cached(name):
    need_login()
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", name): abort(404)
    return send_from_directory(CACHE, name)

@app.route("/api/delete/<int:mid>", methods=["POST"])
def delete(mid):
    need_login()
    r = db().execute("SELECT stored, orig, user_id FROM media WHERE id=?", (mid,)).fetchone()
    if not r or (r["user_id"] != me() and not is_admin()): abort(403)
    db().execute("DELETE FROM media WHERE id=?", (mid,)); db().commit()
    purge(r["stored"])
    audit("delete", f'{r["orig"]} (owner id {r["user_id"]})')
    return jsonify(ok=True)

@app.route("/api/delete_series", methods=["POST"])
def delete_series():
    need_login()
    name = request.form.get("name", "")
    uid = request.form.get("uid", type=int) if is_admin() else None
    uid = uid or me()
    rows = db().execute("SELECT id, stored FROM media WHERE series=? AND user_id=?", (name, uid)).fetchall()
    db().execute("DELETE FROM media WHERE series=? AND user_id=?", (name, uid)); db().commit()
    for r in rows: purge(r["stored"])
    audit("delete_series", f"{name} ({len(rows)} files, owner id {uid})")
    return jsonify(ok=True, n=len(rows))

# ---------- MKV/MOV -> MP4 (stream copy, cached, runs in the background) ----------
def remux_job(mid, src, out):
    tmp = out + ".part.mp4"
    try:
        with CONVERT_SLOT:
            for ac in ("copy", "aac"):          # keep the audio if the browser can play it, else re-encode just audio
                p = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0", "-map", "0:a:0?",
                                    "-c:v", "copy", "-c:a", ac, "-movflags", "+faststart", tmp],
                                   capture_output=True, timeout=4 * 3600)
                if p.returncode == 0:
                    os.replace(tmp, out)
                    with JLOCK: JOBS.pop(mid, None)
                    return
        raise RuntimeError("ffmpeg failed")
    except Exception as e:
        log.warning("remux %s failed: %s", mid, e)
        with JLOCK: JOBS[mid] = "failed"
        try: os.remove(tmp)
        except OSError: pass

@app.route("/api/remux/<int:mid>", methods=["POST"])
def remux(mid):
    need_login()
    r = db().execute("SELECT stored FROM media WHERE id=?", (mid,)).fetchone()
    if not r or not FFMPEG: abort(404)
    stem = r["stored"].rsplit(".", 1)[0]; out = os.path.join(CACHE, stem + ".mp4")
    if os.path.exists(out): return jsonify(state="ready", url="/cache/" + stem + ".mp4")
    with JLOCK:
        st = JOBS.get(mid)
        if st == "failed": JOBS.pop(mid); return jsonify(state="failed")
        if st is None:
            JOBS[mid] = "working"
            threading.Thread(target=remux_job, args=(mid, os.path.join(UPLOADS, r["stored"]), out), daemon=True).start()
    return jsonify(state="working")

# ---------- chunked upload (parallel-safe, retry-safe, resumable) ----------
@app.route("/api/upload/start", methods=["POST"])
def up_start():
    need_login()
    size = request.form.get("size", 0, type=int)
    if size <= 0 or size > MAX_FILE: return jsonify(error="Invalid file size."), 400
    q = quota_bytes(g.user)
    if q:
        used = db().execute("SELECT COALESCE(SUM(size),0) FROM media WHERE user_id=?", (me(),)).fetchone()[0]
        pend = sum(x["size"] for x in UP.values() if x["user"] == me())
        if used + pend + size > q: return jsonify(error="Storage quota exceeded."), 403
    if sum(1 for s in UP.values() if s["user"] == me()) >= MAX_SESSIONS_PER_USER:
        return jsonify(error="Too many unfinished uploads. Try again later."), 429
    if shutil.disk_usage(DATA).free < size + 2 * 1024**3:
        return jsonify(error="Not enough free disk space on the server."), 507
    sweep()
    uid = uuid.uuid4().hex; path = os.path.join(TMP, uid); open(path, "wb").close()
    UP[uid] = {"user": me(), "size": size, "path": path, "lock": threading.Lock(),
               "got": {}, "orig": request.form.get("orig", "file").replace("\\", "/").strip("/").replace("/", " - ")[:200] or "file",
               "series": request.form.get("series", "").strip().replace("/", " ")[:120] or None}
    return jsonify(id=uid)

def sess(uid):
    s = UP.get(uid)
    if not s or s["user"] != me(): abort(404)
    return s

@app.route("/api/upload/chunk/<uid>", methods=["POST"])
def up_chunk(uid):
    need_login(); s = sess(uid)
    off = request.args.get("offset", -1, type=int)
    n = request.content_length or 0
    if off < 0 or n <= 0 or off + n > s["size"]:
        return jsonify(error="Bad chunk."), 400
    left = n                              # stream straight to disk at its own offset:
    with open(s["path"], "r+b") as f:     # no RAM buffer, order doesn't matter, a re-sent chunk overwrites itself
        f.seek(off)
        while left:
            b = request.stream.read(min(left, 1 << 20))
            if not b: break
            f.write(b); left -= len(b)
    if left: return jsonify(error="Short chunk."), 400
    with s["lock"]: s["got"][off] = n
    return jsonify(ok=True)

@app.route("/api/upload/status/<uid>")
def up_status(uid):
    need_login(); s = sess(uid)
    return jsonify(got=[[o, n] for o, n in s["got"].items()])

def complete(got, size):
    pos = 0
    for o in sorted(got):
        if o != pos: return False         # gap or overlap
        pos += got[o]
    return pos == size

@app.route("/api/upload/finish/<uid>", methods=["POST"])
def up_finish(uid):
    need_login(); s = sess(uid)
    if not complete(s["got"], s["size"]):
        return jsonify(error="Upload incomplete."), 400
    with open(s["path"], "rb") as f: ext = sniff(f.read(64))
    if ext not in KIND:
        os.remove(s["path"]); UP.pop(uid, None)
        return jsonify(error="Unsupported type (HEIC/AVIF photos are not supported). Use MP4, MOV, WEBM, MKV, MP3, M4A, WAV, OGG, FLAC, JPG, PNG, GIF or WEBP."), 400
    name = f"{uuid.uuid4().hex}.{ext}"; dest = os.path.join(UPLOADS, name)
    shutil.move(s["path"], dest); UP.pop(uid, None)
    try:
        db().execute("INSERT INTO media(user_id,stored,orig,size,kind,ext,created,series) VALUES(?,?,?,?,?,?,?,?)",
                     (me(), name, s["orig"], s["size"], KIND[ext], ext, time.time(), s["series"]))
        db().commit()
    except sqlite3.Error:
        log.exception("db insert failed"); os.remove(dest)
        return jsonify(error="Database error."), 500
    audit("upload", f'{s["orig"]} ({s["size"]} bytes) -> {name}')
    threading.Thread(target=make_thumb, args=(name, KIND[ext]), daemon=True).start()
    return jsonify(ok=True)

# ---------- admin ----------
@app.route("/api/admin/users")
def adm_users():
    need_admin()
    rows = db().execute("""SELECT u.id, u.name, u.is_admin, u.quota_gb, COUNT(m.id) AS n,
        COALESCE(SUM(m.size),0) AS used FROM users u LEFT JOIN media m ON m.user_id=u.id
        GROUP BY u.id ORDER BY u.id""").fetchall()
    return jsonify(users=[dict(r) for r in rows], default_quota=DEFAULT_QUOTA_GB, invite=INVITE,
                   import_dir=IMPORT_DIR, import_mode=IMPORT_MODE)

@app.route("/api/admin/log")
def adm_log():
    need_admin()
    rows = db().execute("SELECT ts, user, ip, action, detail FROM audit ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/admin/quota", methods=["POST"])
def adm_quota():
    need_admin()
    uid = request.form.get("uid", type=int); raw = request.form.get("gb", "").strip()
    try: val = None if raw == "" else float(raw)
    except ValueError: return jsonify(error="Enter a number."), 400
    if val is not None and val < 0: return jsonify(error="Must be 0 or more."), 400
    db().execute("UPDATE users SET quota_gb=? WHERE id=?", (val, uid)); db().commit()
    audit("set_quota", f"user id {uid} -> {raw or 'default'} GB")
    return jsonify(ok=True)

@app.route("/api/admin/delete_user", methods=["POST"])
def adm_del_user():
    need_admin()
    uid = request.form.get("uid", type=int)
    if uid == me(): return jsonify(error="You cannot delete yourself."), 400
    u = db().execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
    if not u: abort(404)
    rows = db().execute("SELECT stored FROM media WHERE user_id=?", (uid,)).fetchall()
    db().execute("DELETE FROM media WHERE user_id=?", (uid,)); db().execute("DELETE FROM users WHERE id=?", (uid,)); db().commit()
    for r in rows: purge(r["stored"])
    for k, x in list(UP.items()):
        if x["user"] == uid:
            UP.pop(k, None)
            try: os.remove(x["path"])
            except OSError: pass
    audit("delete_user", f'{u["name"]} (id {uid}, {len(rows)} files)')
    return jsonify(ok=True)

# ---------- admin: import files from a folder on the server ----------
def import_job(uid, uname, ip):
    done = skipped = 0; msg = ""
    try:
        c = sqlite3.connect(DB, timeout=10)
        linked = set()
        if IMPORT_MODE == "link":
            for f in os.listdir(UPLOADS):
                p = os.path.join(UPLOADS, f)
                if os.path.islink(p): linked.add(os.path.realpath(p))
        for root, dirs, files in os.walk(IMPORT_DIR):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            rel = os.path.relpath(root, IMPORT_DIR)
            parts = [] if rel == "." else rel.split(os.sep)
            series = parts[0][:120] if parts else None                     # first subfolder = series name
            sub = " - ".join(parts[1:])                                    # deeper folders (Season 1) stay in the name
            for fn in sorted(files):
                src = os.path.join(root, fn); dest = None
                try:
                    if fn.startswith(".") or time.time() - os.path.getmtime(src) < 10:
                        skipped += 1; continue                               # hidden, or still being copied in
                    real = os.path.realpath(src)
                    if IMPORT_MODE == "link" and real in linked: continue    # already imported
                    with open(src, "rb") as f: ext = sniff(f.read(64))
                    if ext not in KIND: skipped += 1; continue
                    size = os.path.getsize(src)
                    name = f"{uuid.uuid4().hex}.{ext}"; dest = os.path.join(UPLOADS, name)
                    if IMPORT_MODE == "link": os.symlink(real, dest); linked.add(real)
                    else: shutil.move(src, dest)
                    c.execute("INSERT INTO media(user_id,stored,orig,size,kind,ext,created,series) VALUES(?,?,?,?,?,?,?,?)",
                              (uid, name, ((sub + " - " if sub else "") + fn)[:200], size, KIND[ext], ext, time.time(), series)); c.commit()
                    threading.Thread(target=make_thumb, args=(name, KIND[ext]), daemon=True).start()
                    done += 1; IMPORT["done"] = done
                except Exception:
                    log.exception("import failed for %s", src); skipped += 1
                    if dest and os.path.lexists(dest) and not os.path.exists(src):
                        pass                                                 # moved but DB failed: keep the file
                    elif dest and os.path.lexists(dest):
                        try: os.remove(dest)
                        except OSError: pass
        if IMPORT_MODE == "move":                                            # tidy empty folders left behind
            for root, dirs, files in os.walk(IMPORT_DIR, topdown=False):
                if root != IMPORT_DIR and not os.listdir(root):
                    try: os.rmdir(root)
                    except OSError: pass
        c.execute("INSERT INTO audit(ts,user_id,user,ip,action,detail) VALUES(?,?,?,?,?,?)",
                  (time.time(), uid, uname, ip, "import", f"{done} added, {skipped} skipped ({IMPORT_MODE})")); c.commit(); c.close()
    except Exception as e:
        log.exception("import crashed"); msg = "error: " + str(e)[:80]
    with ILOCK: IMPORT.update(state="idle", done=done, skipped=skipped, msg=msg)

@app.route("/api/admin/import", methods=["POST"])
def adm_import():
    need_admin()
    with ILOCK:
        if IMPORT["state"] != "running":
            IMPORT.update(state="running", done=0, skipped=0, msg="")
            threading.Thread(target=import_job, args=(me(), session.get("name"), client_ip()), daemon=True).start()
        return jsonify(dict(IMPORT))

@app.route("/api/admin/import/status")
def adm_import_status():
    need_admin(); return jsonify(dict(IMPORT))

# ---------- speed test ----------
BLOCK = os.urandom(1024 * 1024)

@app.route("/api/speedtest/down")
def st_down():
    need_login()
    mb = max(1, min(request.args.get("mb", 16, type=int), 256))
    return Response((BLOCK for _ in range(mb)), mimetype="application/octet-stream",
                    headers={"Content-Length": str(mb * 1024 * 1024), "Cache-Control": "no-store"})

@app.route("/api/speedtest/up", methods=["POST"])
def st_up():
    need_login(); n = 0
    while True:
        b = request.stream.read(1 << 20)
        if not b: break
        n += len(b)
    return jsonify(bytes=n)

@app.route("/api/speedtest/disk", methods=["POST"])
def st_disk():
    need_login()
    p = os.path.join(TMP, "disk_" + secrets.token_hex(4)); t = time.time()
    with open(p, "wb") as f:
        for _ in range(64): f.write(BLOCK)
        f.flush(); os.fsync(f.fileno())
    dt = max(time.time() - t, 1e-6); os.remove(p)
    return jsonify(mbps=64 / dt)

# ---------- templates ----------
CSS = r"""
:root{--navy:#102343;--blue:#2f4d80;--accent:#fdbd27;--page:#f5f7fb;--text:#102343;--muted:#7890b2;--line:#e4eaf3;--green:#159a72;--danger:#d74b4b}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--text);font:15px "DM Sans",system-ui,Arial,sans-serif;
background:radial-gradient(circle at 92% -12%,#e4edff 0,transparent 27%),var(--page)}
button,input,select{font:inherit}button{cursor:pointer}
:focus-visible{outline:2px solid #3977d7;outline-offset:2px}
.btn{background:linear-gradient(135deg,#3977d7,#245398);color:#fff;border:0;border-radius:9px;padding:10px 16px;font-weight:700;transition:transform .18s,box-shadow .18s}
.btn:hover{transform:translateY(-1px);box-shadow:0 7px 14px rgba(28,67,124,.18)}
.btn:disabled{opacity:.5;transform:none}
.btn.alt{background:#fff;color:var(--text);border:1px solid var(--line)}
.btn.alt:hover{border-color:#9ab8e8;color:#1553ad}.btn.del:hover{color:var(--danger);border-color:var(--danger)}
input,select{width:100%;padding:11px 13px;border:1.5px solid #d9e2f0;border-radius:10px;background:#f9fbff}
input:focus,select:focus{border-color:#2965bd;box-shadow:0 0 0 3px rgba(41,101,189,.15);outline:0}
.err{background:#ffeaea;color:#c0392b;border:1px solid #f5c6cb;border-radius:8px;padding:11px 14px;margin-bottom:16px;font-weight:600;font-size:13px}
.logo{width:44px;height:44px;border-radius:12px;background:var(--accent);display:grid;place-items:center;flex:none}
.logo:after{content:"";border-left:14px solid var(--navy);border-top:9px solid transparent;border-bottom:9px solid transparent;margin-left:4px}
"""

LOGIN = r"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Sign in | Photogram</title>
<style>__CSS__
body{display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0b1e3d,#1a3a6b 50%,#102343)}
.card{width:100%;max-width:410px;margin:20px;background:#fff;border-radius:22px;padding:40px 34px;box-shadow:0 30px 70px rgba(0,0,0,.35)}
.top{display:flex;gap:14px;align-items:center;margin-bottom:26px}.top strong{display:block;font-size:22px}
.top span{font-size:11px;letter-spacing:1.4px;color:var(--muted)}
h2{margin:0 0 6px;font-size:24px}p{color:var(--muted);margin:0 0 22px;font-size:14px}
label{display:block;margin-bottom:16px;font-size:13px;font-weight:600;color:#566f92}label input{margin-top:6px}
.btn{width:100%;padding:14px;font-size:16px}</style>
<div class=card><div class=top><div class=logo></div><div><strong>Photogram</strong><span>MEDIA LIBRARY</span></div></div>
<h2>Sign in</h2><p>New name? Enter the invite code and choose a PIN to create your account.</p>__ERR__
<form method=post><input type=hidden name=csrf value="__CSRF__">
<label>Name<input name=name maxlength=24 autocomplete=username required></label>
<label>PIN (4 to 8 digits)<input name=pin type=password inputmode=numeric maxlength=8 autocomplete=current-password required></label>
<label>Invite code (new accounts only)<input name=invite autocomplete=off></label>
<button class=btn>Sign in</button></form></div></html>""".replace("__CSS__", CSS)

PAGE = r"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Photogram</title>
<style>__CSS__
.side{position:fixed;inset:0 auto 0 0;width:270px;background:var(--navy);color:#e8f0ff;display:flex;flex-direction:column}
.brand{display:flex;gap:12px;align-items:center;padding:22px 18px;border-bottom:1px solid rgba(255,255,255,.1)}
.brand strong{display:block;color:#fff;font-size:19px}.brand span{font-size:10px;letter-spacing:1.3px;color:#a8c4ed}
nav{padding:16px 12px;flex:1}
nav a{display:block;padding:12px 14px;margin-bottom:4px;border-radius:9px;border-left:3px solid transparent;cursor:pointer;transition:.2s}
nav a:hover{background:#1a3763;border-left-color:#ffd45d;transform:translateX(3px)}
nav a.on{background:var(--blue);border-left-color:var(--accent);color:#fff}
.profile{padding:16px;border-top:1px solid rgba(255,255,255,.12);font-size:13px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.profile button{white-space:nowrap;background:none;border:1px solid rgba(255,255,255,.25);color:#e8f0ff;border-radius:8px;padding:6px 10px}
main{margin-left:270px;padding:28px 38px 60px;max-width:1180px}
h1{margin:0 0 18px;font-size:26px}.view{display:none}.view.on{display:block}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-bottom:16px}
.metrics .card{margin:0}.metrics small,.lbl{color:var(--muted);font-size:13px}.metrics b{display:block;font-size:24px;margin-top:4px}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.tools input{flex:1;min-width:160px}
.tabs{display:flex;gap:8px;flex-wrap:wrap}
.tabs button{border:1px solid var(--line);background:#fff;border-radius:9px;padding:9px 14px;color:var(--text)}
.tabs button.on{background:linear-gradient(135deg,#3977d7,#245398);color:#fff;border-color:transparent}
.row{display:flex;gap:12px;align-items:center;padding:12px 8px;border-bottom:1px solid var(--line)}
.row:last-child{border:0}.row:hover{background:#f2f6ff;box-shadow:inset 3px 0 #4b82dd}
.badge{flex:none;width:56px;height:56px;border-radius:10px;display:grid;place-items:center;font-weight:800;font-size:13px;letter-spacing:.5px;position:relative;overflow:hidden}
.badge img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.k-video{background:var(--navy);color:var(--accent)}.k-audio{background:#2965bd;color:#fff}.k-image{background:var(--green);color:#fff}
.meta{min-width:0;flex:1}.meta b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.meta small{color:var(--muted)}
.acts{display:flex;gap:6px;flex:none}.acts .btn{padding:8px 12px;font-size:13px}
.empty{padding:34px 10px;text-align:center;color:var(--muted)}
.drop{border:2px dashed #b9c9e3;border-radius:14px;padding:34px 16px;text-align:center;color:var(--muted);background:#fff;cursor:pointer}
.drop.over{border-color:#3977d7;background:#edf4ff}.drop b{color:var(--text);display:block;font-size:17px;margin-bottom:4px}
.opts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.opts label{font-size:13px;font-weight:600;color:#566f92}.opts select{margin-top:6px}
.bar{height:8px;background:#e4eaf3;border-radius:6px;overflow:hidden;margin-top:8px}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,#3977d7,var(--accent));transition:width .3s}
.q{padding:10px 0;border-bottom:1px solid var(--line)}.q:last-child{border:0}.q .top{display:flex;gap:10px;justify-content:space-between}
.q .top span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.q small{color:var(--muted);white-space:nowrap}
.q.error small{color:var(--danger)}.q.done small{color:var(--green)}
.big{font-size:32px;font-weight:800;line-height:1.1}.big small{font-size:14px;font-weight:600;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}
.grid .card{margin:0}table{width:100%;border-collapse:collapse}td,th{padding:9px 6px;text-align:left;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.note{color:var(--muted);font-size:13px;line-height:1.5}
.modal{position:fixed;inset:0;z-index:9;background:rgba(16,35,67,.6);backdrop-filter:blur(3px);display:none;align-items:center;justify-content:center;padding:16px}
.modal.open{display:flex}.mc{background:#fff;border-radius:16px;width:100%;max-width:760px;max-height:92vh;overflow:auto;padding:18px}
.mh{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.mh b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mh button{background:none;border:0;font-size:26px;color:var(--muted)}
.mc video,.mc img{width:100%;max-height:70vh;background:#000;border-radius:10px;object-fit:contain}.mc audio{width:100%}
.pbar{display:flex;gap:8px;align-items:center;justify-content:center;margin-top:12px;flex-wrap:wrap}
.pbar .btn{padding:8px 14px}.pbar select{width:auto;padding:8px 10px}.pbar span{color:var(--muted);font-size:13px}
.k-series{background:#e4edff;color:var(--blue)}.row.ser{cursor:pointer}
.crumb{display:flex;gap:10px;align-items:center;margin-bottom:12px}.crumb b{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:17px}
nav a i{font-style:normal;display:inline-block;width:22px;margin-right:12px;text-align:center}.dk{display:block}
h1{font-size:32px}.sub{margin:-10px 0 20px;color:var(--muted)}
.metrics .card{display:flex;gap:14px;align-items:center;padding:20px}.ico{display:grid;place-items:center;width:50px;height:50px;border-radius:13px;font-style:normal;font-size:21px;flex:none}
.c1{color:#2873e5;background:#e9f1ff}.c2{color:#ec9813;background:#fff3de}.c3{color:#7a5bf1;background:#f0eaff}
.profile .av{display:grid;place-items:center;width:34px;height:34px;border-radius:50%;background:var(--accent);color:var(--navy);font-weight:800;flex:none}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px}
.tile{border:1.5px solid var(--line);background:#fff;border-radius:14px;padding:20px 8px;text-align:center;color:var(--text);transition:.18s}
.tile:hover{border-color:#3977d7;transform:translateY(-2px);box-shadow:0 8px 18px rgba(28,67,124,.12)}
.tile .ico{margin:0 auto 10px;width:52px;height:52px}.tile b{display:block}.tile small{color:var(--muted);font-size:12px}
.ovl{display:flex;justify-content:space-between;gap:10px;margin-bottom:4px}
.mc .badge{width:100px;height:100px;font-size:22px;margin:10px auto 18px}
@media(max-width:800px){.side{position:static;width:auto;flex-direction:row}
.brand{border:0;padding:12px 14px}.profile{margin-left:auto;border:0}main{margin:0;padding:16px 14px 96px}
nav{position:fixed;left:0;right:0;bottom:0;z-index:6;display:flex;background:var(--navy);padding:6px 4px calc(6px + env(safe-area-inset-bottom));border-top:1px solid rgba(255,255,255,.12)}
nav a{flex:1;text-align:center;font-size:11px;margin:0;padding:6px 2px;border:0;border-radius:10px;color:#a8c4ed}
nav a:hover{transform:none;background:none}nav a.on{background:none;color:var(--accent)}
nav a i{display:block;font-size:19px;line-height:1.2;width:auto;margin:0}nav a.dk,.tabs{display:none}
.row{flex-wrap:wrap}.acts{width:100%;justify-content:flex-end}
.metrics{grid-template-columns:repeat(3,1fr);gap:8px}.metrics .card{padding:12px}.metrics b{font-size:18px}.ico{display:none}.tile .ico{display:grid}
.tiles{grid-template-columns:1fr 1fr}h1{font-size:26px}nav a{font-size:10px!important}
.crumb{flex-wrap:wrap}.crumb b{order:-1;flex-basis:100%;white-space:normal}.crumb .btn{padding:8px 12px;font-size:13px}}
</style>
<aside class=side><div class=brand><div class=logo></div><div><strong>Photogram</strong><span>MEDIA LIBRARY</span></div></div>
<nav><a data-v=lib data-k=all class="dk on"><i>&#9638;</i>Library</a><a data-v=lib data-k=video><i>&#9654;</i>Video</a><a data-v=lib data-k=audio><i>&#9835;</i>Music</a><a data-v=lib data-k=image><i>&#9635;</i>Photos</a><a data-v=lib data-k=series><i>&#9776;</i>Series</a><a data-v=up><i>&#8679;</i>Upload</a><a data-v=speed><i>&#9889;</i>Speed</a>__ADMINNAV__</nav>
<form class=profile method=post action=/logout><input type=hidden name=csrf value="__CSRF__"><span style="display:flex;gap:10px;align-items:center"><span class=av>__INIT__</span>__NAME__</span><button>Log out</button></form></aside>
<main>
<section id=lib class="view on"><h1 id=ttl>Library</h1><p class=sub id=sub></p>
<div class=metrics><div class=card><i class="ico c1">&#9654;</i><div><small>Files</small><b id=m1>0</b></div></div><div class=card><i class="ico c2">&#9635;</i><div><small>Library size</small><b id=m2>0</b></div></div><div class=card><i class="ico c3">&#9776;</i><div><small>Free on server</small><b id=m3>0</b></div></div></div>
<div class=tools><input id=q placeholder="Search files" type=search><div class=tabs id=tabs></div></div>
<div class=card id=list></div></section>

<section id=up class=view><h1>Upload</h1>
<div class=tiles>
<button class=tile data-pick=fVideo><i class="ico c1">&#9654;</i><b>Video</b><small>MP4 MOV WEBM MKV</small></button>
<button class=tile data-pick=fMusic><i class="ico c2">&#9835;</i><b>Music</b><small>MP3 M4A WAV FLAC OGG</small></button>
<button class=tile data-pick=fPics><i class="ico c3">&#9635;</i><b>Pictures</b><small>JPG PNG GIF WEBP</small></button>
<button class=tile id=dirbtn><i class="ico" style="color:#e54f50;background:#ffebeb">&#9776;</i><b>Series folder</b><small>Whole show or album</small></button></div>
<div class=drop id=drop style="padding:16px"><b style="font-size:15px">Or drop files here</b>Up to 50 GB each</div>
<div class=card style="margin-top:14px"><label class=lbl style="font-weight:600">Series / album name (optional)<input id=sname maxlength=100 placeholder="Group the files you pick into one series" style="margin-top:6px"></label>
<p class=note style="margin:10px 0 0">A folder becomes one series automatically (subfolders like Season 1 stay in the episode name, so they sort in order). Non-media files are skipped. If a transfer breaks, add the same file again and it resumes.</p></div>
<p class=note id=quota style="text-align:center"></p>
<input type=file id=files multiple hidden accept="video/*,audio/*,image/*,.mkv,.flac">
<input type=file id=fVideo multiple hidden accept="video/*,.mkv,.mov"><input type=file id=fMusic multiple hidden accept="audio/*,.flac,.m4a">
<input type=file id=fPics multiple hidden accept="image/*"><input type=file id=dirs hidden webkitdirectory multiple>
<div class=opts><label>Streams at once<select id=streams><option>1<option>2<option selected>3<option>4</select></label>
<label>Chunk size<select id=chunk><option value=4>4 MB<option value=8>8 MB<option value=16 selected>16 MB<option value=32>32 MB</select></label></div>
<p class=note id=info style="margin:-4px 0 14px"></p>
<div class=card><div class=grid>
<div><span class=lbl>Speed now</span><div class=big id=spd>0 <small>MB/s</small></div><span class=lbl id=mbps>0 Mbps</span></div>
<div><span class=lbl>Time left</span><div class=big id=eta>--</div><span class=lbl id=avg>average --</span></div>
<div><span class=lbl>Sent</span><div class=big id=sentv>0 <small>MB</small></div><span class=lbl id=peak>peak --</span></div></div></div>
<div class=card id=queue><div class=empty>Nothing uploading.</div></div></section>

<section id=speed class=view><h1>Speed test</h1>
<div class=card><p class=note style="margin-top:0">Measures the same path your uploads use (this device to the server, with your "Streams at once" setting). Disk write shows how fast the server can store data.</p>
<button class=btn id=run>Run test</button> <span class=note id=st></span></div>
<div class=grid id=res></div>
<div class=card id=eta5 style="margin-top:16px;display:none"><b>Time to upload</b><table id=tbl></table></div>
<div class=card><b>How to read it</b><p class=note>Upload speed is limited by the slowest of three things: the Wi-Fi link, the server's storage, and the server's CPU. If disk write is near your upload speed, storage is the limit. If disk is much faster, Wi-Fi is. When the phone uploading and the phone serving share one router, every byte crosses the air twice, so you usually get about half the link rate. 5 GHz Wi-Fi, being close to the router, USB tethering, or a cable on the laptop side all help.</p></div></section>
<section id=adm class=view><h1>Admin</h1>
<div class=card><b>Users</b><p class=note id=invtxt></p><div id=users></div></div>
<div class=card><b>Import from folder</b><p class=note id=impnote></p>
<button class=btn id=impbtn>Scan folder now</button> <span class=note id=impst></span></div>
<div class=card><b>Activity log</b> <span class=note>(last 200)</span><div style="overflow:auto"><table id=logt></table></div></div></section>
</main>
<div class=modal id=modal><div class=mc><div class=mh><b id=mt></b><button id=mx aria-label=Close>&times;</button></div><div id=mb></div>
<div class=pbar id=pbar><button class="btn alt" id=pv>&#9198; Prev</button><select id=rate aria-label=Speed><option value=0.5>0.5x<option value=0.75>0.75x<option value=1 selected>1x<option value=1.25>1.25x<option value=1.5>1.5x<option value=2>2x</select><button class="btn alt" id=nx>Next &#9197;</button></div>
<div class=pbar><span>Space play/pause &middot; Arrows seek/volume &middot; F fullscreen &middot; N / P next / prev</span></div></div></div>
<script>
const CSRF="__CSRF__",$=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const B=n=>n>=1e9?(n/1e9).toFixed(2)+' GB':n>=1e6?(n/1e6).toFixed(1)+' MB':Math.max(1,Math.round(n/1e3))+' KB';
const T=s=>{if(!isFinite(s)||s<0)return '--';s=Math.round(s);const h=s/3600|0,m=s%3600/60|0;return h?h+'h '+m+'m':m?m+'m '+s%60+'s':s+'s'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function api(u,o={}){o.headers={...(o.headers||{}),'X-CSRF':CSRF};const r=await fetch(u,o);
 if(r.status===401){location='/login';throw new Error('auth')}return r.json()}
function xhr(url,blob,onp){return new Promise((res,rej)=>{const x=new XMLHttpRequest();x.open('POST',url);
 x.setRequestHeader('X-CSRF',CSRF);x.setRequestHeader('Content-Type','application/octet-stream');
 x.upload.onprogress=e=>onp(e.loaded);
 x.onload=()=>{if(x.status===200){onp(blob.size);res(JSON.parse(x.responseText))}else rej(new Error(x.status))};
 x.onerror=x.ontimeout=()=>rej(new Error('net'));x.send(blob)})}

function go(v,k){if(k){FILTER=k;SER=null}
 $$('nav a').forEach(x=>x.classList.toggle('on',x.dataset.v===v&&(v!=='lib'||x.dataset.k===FILTER)));
 $$('.view').forEach(x=>x.classList.toggle('on',x.id===v));if(v==='lib')load();if(v==='adm')loadAdm()}
$$('nav a').forEach(a=>a.onclick=()=>go(a.dataset.v,a.dataset.k));

/* ---- library ---- */
let MEDIA=[],ST={used:0,free:0},FILTER='all',SER=null;
const sk=m=>m.user_id+'|'+m.series,nm=k=>k.slice(k.indexOf('|')+1),SERN=()=>nm(SER);
const nat=(a,b)=>a.orig.localeCompare(b.orig,undefined,{numeric:true,sensitivity:'base'});
const KL={all:'All',video:'Video',audio:'Music',image:'Photos',series:'Series'};
async function load(){[MEDIA,ST]=await Promise.all([api('/api/media'),api('/api/stats')]);render()}
function render(){
 const q=$('#q').value.toLowerCase();
 $('#m1').textContent=MEDIA.length;$('#m2').textContent=B(ST.used);$('#m3').textContent=B(ST.free);
 $('#quota').textContent='Your storage: '+B(ST.mine)+(ST.quota?' of '+B(ST.quota):' (no limit)');
 $('#tabs').innerHTML=Object.keys(KL).map(k=>{const n=k==='all'?MEDIA.length:k==='series'?new Set(MEDIA.filter(m=>m.series).map(sk)).size:MEDIA.filter(m=>m.kind===k).length;
  return `<button data-k=${k} class="${k===FILTER?'on':''}">${KL[k]} ${n}</button>`}).join('');
 $$('#tabs button').forEach(b=>b.onclick=()=>go('lib',b.dataset.k));
 $$('nav a[data-v=lib]').forEach(a=>a.classList.toggle('on',a.dataset.k===FILTER));
 const L=vis(),flat=q||SER;
 $('#ttl').textContent=SER?SERN():KL[FILTER]==='All'?'Library':KL[FILTER];
 $('#sub').textContent=SER?'Plays in order, next episode starts automatically':FILTER==='series'?'Shows, albums and folders':new Date().toLocaleDateString([], {weekday:'long',year:'numeric',month:'long',day:'numeric'});
 const fileRow=m=>`<div class=row>
  <div class="badge k-${m.kind}">${m.ext.toUpperCase()}${m.kind!=='audio'?`<img src="/thumb/${m.stored}" loading=lazy alt="" onerror="this.remove()">`:''}</div>
  <div class=meta><b>${esc(m.orig)}</b><small>${B(m.size)} &middot; ${new Date(m.created*1000).toLocaleString([], {dateStyle:'medium',timeStyle:'short'})} &middot; ${esc(m.owner)}${m.series&&!SER?' &middot; '+esc(m.series):''}</small></div>
  <div class=acts><button class=btn onclick="play(${m.id})">Play</button>
  <a class="btn alt" style="text-decoration:none" href="/media/${m.stored}?dl=1">Save</a>
  ${m.mine?`<button class="btn alt del" onclick="del(${m.id})">Delete</button>`:''}</div></div>`;
 let h='';
 if(SER){const own=L.some(m=>m.mine);
  h=`<div class=crumb><button class="btn alt" onclick="SER=null;render()">&larr; Back</button><b>${esc(SERN())}</b><button class=btn onclick="playAll()">&#9654; Play all</button>${own?`<button class="btn alt del" onclick="delSer()">Delete series</button>`:''}</div>`+
   (L.length?L.map(fileRow).join(''):'<div class=empty>This series is empty.</div>')}
 else if(q){h=L.length?L.map(fileRow).join(''):'<div class=empty>No matches.</div>'}
 else{const g={},loose=[];L.forEach(m=>m.series?(g[sk(m)]=g[sk(m)]||[]).push(m):loose.push(m));
  h=Object.keys(g).sort((a,b)=>nm(a).localeCompare(nm(b),undefined,{numeric:true})).map(n=>{const a=g[n];
   return `<div class="row ser" data-s="${esc(n)}"><div class="badge k-series">${a.length}</div>
   <div class=meta><b>${esc(nm(n))}</b><small>${a.length} file${a.length>1?'s':''} &middot; ${B(a.reduce((t,m)=>t+m.size,0))} &middot; ${esc(a[0].owner)}</small></div>
   <div class=acts><button class=btn>Open</button></div></div>`}).join('')+(FILTER==='series'?'':loose.map(fileRow).join(''));
  if(!h)h=FILTER==='series'?'<div class=empty>No series yet. Open Upload and choose Series folder.</div>':'<div class=empty>Nothing here yet. Open Upload to add files.</div>'}
 $('#list').innerHTML=h;
 $$('#list .ser').forEach(r=>r.onclick=()=>{SER=r.dataset.s;render();scrollTo(0,0)})}
$('#q').oninput=render;
let CUR=null;
function vis(){const q=$('#q').value.toLowerCase();let L=MEDIA.filter(m=>(FILTER==='all'||m.kind===FILTER||(FILTER==='series'&&m.series))&&(m.orig.toLowerCase().includes(q)||(m.series||'').toLowerCase().includes(q)));
 if(SER)L=L.filter(m=>sk(m)===SER).sort(nat);return L}
function step(d){const L=vis(),i=L.findIndex(m=>m.id===CUR);if(i<0||!L.length)return;play(L[(i+d+L.length)%L.length].id)}
function playAll(){const L=vis();if(L.length)play(L[0].id)}
function media(){return $('#mb video,#mb audio')}
async function play(id){CUR=id;const m=MEDIA.find(x=>x.id===id);let u='/media/'+m.stored;
 $('#mt').textContent=m.orig+' ('+m.ext.toUpperCase()+')';$('#modal').classList.add('open');
 if(m.kind==='video'&&(m.ext==='mkv'||m.ext==='mov')&&ST.ffmpeg){
  $('#pbar').style.display='none';$('#mb').innerHTML='<p class=note>Preparing for playback (first time only, large files take a while)...</p>';
  try{let r=await api('/api/remux/'+m.id,{method:'POST'});
   while(r.state==='working'){await sleep(2000);if(CUR!==id)return;r=await api('/api/remux/'+m.id,{method:'POST'})}
   if(CUR!==id)return;if(r.url)u=r.url}catch(e){if(CUR!==id)return}}
 $('#mb').innerHTML=m.kind==='video'?`<video controls autoplay playsinline src="${u}"></video>`
  :m.kind==='audio'?`<div class="badge k-audio">${m.ext.toUpperCase()}</div><audio controls autoplay src="${u}"></audio>`
  :`<img src="${u}" alt="">`;
 if((m.ext==='mkv'||m.ext==='mov')&&u.startsWith('/media/'))$('#mb').insertAdjacentHTML('beforeend','<p class=note>If it will not play, your browser may not support this format. Use Save and open it in VLC.</p>');
 $('#pbar').style.display=m.kind==='image'?'none':'flex';
 const md=media();if(md){md.playbackRate=+$('#rate').value;md.onended=()=>{const L=vis(),i=L.findIndex(m=>m.id===CUR);if(i>=0&&i<L.length-1)step(1)}}}
function closeM(){$('#modal').classList.remove('open');$('#mb').innerHTML='';CUR=null}
$('#pv').onclick=()=>step(-1);$('#nx').onclick=()=>step(1);
$('#rate').onchange=()=>{const md=media();if(md)md.playbackRate=+$('#rate').value};
document.addEventListener('keydown',e=>{if(!$('#modal').classList.contains('open')||/INPUT|SELECT/.test(e.target.tagName))return;
 const md=media(),k=e.key.toLowerCase();
 if(k==='escape')closeM();else if(k==='n')step(1);else if(k==='p')step(-1);
 else if(md){if(k===' '){md.paused?md.play():md.pause()}else if(k==='arrowright')md.currentTime+=10;else if(k==='arrowleft')md.currentTime-=10;
  else if(k==='arrowup')md.volume=Math.min(1,md.volume+.1);else if(k==='arrowdown')md.volume=Math.max(0,md.volume-.1);
  else if(k==='f'&&md.requestFullscreen)md.requestFullscreen();else return;}
 else return;e.preventDefault()});
$('#mx').onclick=closeM;$('#modal').onclick=e=>{if(e.target.id==='modal')closeM()};
async function delSer(){if(confirm('Delete your files in "'+SERN()+'" from the server?')){await api('/api/delete_series',{method:'POST',body:new URLSearchParams({name:SERN(),uid:SER.split('|')[0]})});SER=null;load()}}
async function del(id){if(confirm('Delete this file from the server?')){await api('/api/delete/'+id,{method:'POST'});load()}}

/* ---- admin ---- */
let USERS=[];
async function loadAdm(){const [u,l]=await Promise.all([api('/api/admin/users'),api('/api/admin/log')]);USERS=u.users;
 $('#invtxt').innerHTML='Invite code: <b>'+esc(u.invite)+'</b> &middot; default quota: '+(u.default_quota?u.default_quota+' GB':'none');
 $('#impnote').innerHTML='Put files in <b>'+esc(u.import_dir)+'</b>, then scan. A subfolder becomes a series. Mode: '+(u.import_mode==='link'?'link (files stay where they are)':'move (files are taken into the library)')+'.';
 $('#users').innerHTML=u.users.map(x=>`<div class=row><div class=meta><b>${esc(x.name)}${x.is_admin?' (admin)':''}</b>
  <small>${x.n} files &middot; ${B(x.used)} &middot; quota ${x.is_admin?'unlimited':x.quota_gb==null?'default':x.quota_gb?x.quota_gb+' GB':'unlimited'}</small></div>
  <div class=acts>${x.is_admin?'':`<button class="btn alt" onclick="setQ(${x.id})">Quota</button><button class="btn alt del" onclick="delUser(${x.id})">Delete</button>`}</div></div>`).join('');
 $('#logt').innerHTML='<tr><th>Time<th>User<th>IP<th>Action<th>Detail</tr>'+l.map(r=>`<tr><td>${new Date(r.ts*1000).toLocaleString([], {dateStyle:'short',timeStyle:'short'})}<td>${esc(r.user||'')}<td>${esc(r.ip||'')}<td>${esc(r.action)}<td>${esc(r.detail||'')}</tr>`).join('')}
$('#impbtn').onclick=async()=>{const b=$('#impbtn');b.disabled=true;
 try{let r=await api('/api/admin/import',{method:'POST'});
  while(r.state==='running'){$('#impst').textContent='Importing... '+r.done+' added';await sleep(1500);r=await api('/api/admin/import/status')}
  $('#impst').textContent='Done: '+r.done+' added, '+r.skipped+' skipped'+(r.msg?' ('+r.msg+')':'')}catch(e){$('#impst').textContent='Failed'}
 b.disabled=false;loadAdm()};
async function setQ(id){const v=prompt('Storage quota in GB for this user (0 = unlimited, empty = default):');if(v===null)return;
 const r=await api('/api/admin/quota',{method:'POST',body:new URLSearchParams({uid:id,gb:v})});if(r.error)alert(r.error);loadAdm()}
async function delUser(id){const x=USERS.find(u=>u.id===id);if(!x||!confirm('Delete user "'+x.name+'" and ALL their files? This cannot be undone.'))return;
 const r=await api('/api/admin/delete_user',{method:'POST',body:new URLSearchParams({uid:id})});if(r.error)alert(r.error);loadAdm()}

/* ---- upload + live speed ---- */
let QU=[],running=false,sent=0,samples=[],peak=0,t0=0,meter=null;
$('#drop').onclick=()=>$('#files').click();
['files','fVideo','fMusic','fPics'].forEach(id=>$('#'+id).onchange=e=>{add([...e.target.files],false,$('#sname').value.trim());e.target.value=''});
$$('[data-pick]').forEach(b=>b.onclick=()=>$('#'+b.dataset.pick).click());
$('#dirbtn').onclick=()=>$('#dirs').click();
const OKX=/\.(mp4|mov|webm|mkv|mp3|m4a|wav|ogg|flac|jpg|jpeg|png|gif|webp)$/i;
$('#dirs').onchange=e=>{const fl=[...e.target.files].sort((a,b)=>a.webkitRelativePath.localeCompare(b.webkitRelativePath,undefined,{numeric:true}));
 const ok=fl.filter(f=>OKX.test(f.name));if(fl.length>ok.length)alert((fl.length-ok.length)+' non-media file(s) skipped.');
 add(ok,true,$('#sname').value.trim());e.target.value=''};
['dragover','dragenter'].forEach(ev=>$('#drop').addEventListener(ev,e=>{e.preventDefault();$('#drop').classList.add('over')}));
['dragleave','drop'].forEach(ev=>$('#drop').addEventListener(ev,e=>{e.preventDefault();$('#drop').classList.remove('over')}));
$('#drop').addEventListener('drop',e=>add([...e.dataTransfer.files],false,$('#sname').value.trim()));
const nat2=(a,b)=>a.localeCompare(b,undefined,{numeric:true,sensitivity:'base'});
function add(fl,dir,sname){fl=[...fl];if(sname&&!dir)fl.sort((a,b)=>nat2(a.name,b.name));
 for(const f of fl){const rp=dir&&f.webkitRelativePath?f.webkitRelativePath.split('/'):null;
  QU.push({f,state:'queued',done:0,orig:rp&&rp.length>1?rp.slice(1).join(' - '):f.name,series:sname||(rp?rp[0]:'')})}
 drawQ();if(!running)runQ()}
function drawQ(){
 if(!QU.length){$('#queue').innerHTML='<div class=empty>Nothing uploading.</div>';return}
 const tot=QU.reduce((n,i)=>n+i.f.size,0),got=QU.reduce((n,i)=>n+(i.state==='done'?i.f.size:i.done),0),nd=QU.filter(i=>i.state==='done').length,ne=QU.filter(i=>i.state==='error').length;
 const act=QU.filter(i=>i.state!=='done'),show=act.slice(0,25);
 $('#queue').innerHTML=`<div class=ovl><b>${nd} of ${QU.length} files done${ne?', '+ne+' failed':''}</b><span class=lbl>${B(got)} / ${B(tot)}</span></div><div class=bar style="margin-bottom:8px"><i style="width:${tot?got/tot*100:0}%"></i></div>`+
  show.map(it=>{const p=Math.floor(it.done/it.f.size*100);
  const lab=it.state==='error'?esc(it.err):it.state==='queued'?'Waiting':p+'%';
  return `<div class="q ${it.state}"><div class=top><span>${it.series?esc(it.series)+' / ':''}${esc(it.orig)}</span><small>${B(it.f.size)} &middot; ${lab}</small></div><div class=bar><i style="width:${p}%"></i></div></div>`}).join('')+
  (act.length>25?`<p class=note style="margin:10px 0 0">+ ${act.length-25} more waiting</p>`:'')}
function info(){const CH=+$('#chunk').value*1048576;
 $('#info').innerHTML=`Each file is cut into <b>${$('#chunk').value} MB chunks</b>, about ${Math.round(CH/1460).toLocaleString()} network packets of 1,460 bytes each. A 50 GB file becomes ${Math.ceil(50*1024**3/CH).toLocaleString()} chunks. ${$('#streams').value} chunk(s) travel at once and a failed chunk is retried by itself. A series can hold any number of files.`}
$('#chunk').onchange=info;$('#streams').onchange=info;info();
function tick(){
 const now=performance.now();samples.push([now,sent]);samples=samples.filter(s=>now-s[0]<=4000);
 const a=samples[0],dt=(now-a[0])/1000,bps=dt>0.4?(sent-a[1])/dt:0;peak=Math.max(peak,bps);
 const left=QU.reduce((n,i)=>i.state==='queued'||i.state==='uploading'?n+i.f.size-i.done:n,0),avg=sent/((now-t0)/1000);
 $('#spd').innerHTML=(bps/1e6).toFixed(2)+' <small>MB/s</small>';$('#mbps').textContent=(bps*8/1e6).toFixed(1)+' Mbps';
 $('#eta').textContent=bps>1e4?T(left/bps):'--';$('#avg').textContent='average '+(avg/1e6).toFixed(2)+' MB/s';
 $('#sentv').innerHTML=(sent/1e6).toFixed(0)+' <small>MB</small>';$('#peak').textContent='peak '+(peak/1e6).toFixed(2)+' MB/s';drawQ()}
async function runQ(){running=true;sent=0;peak=0;samples=[];t0=performance.now();meter=setInterval(tick,500);
 for(const it of QU)if(it.state==='queued')await one(it);
 running=false;clearInterval(meter);tick();load()}
async function one(it){
 it.state='uploading';const f=it.f,N=+$('#streams').value;let CH=+$('#chunk').value*1048576;
 try{
 const fp='up:'+[f.name,f.size,f.lastModified,it.series].join('|'),have=new Set();let st=null;
 try{const sv=JSON.parse(localStorage.getItem(fp)||'null');if(sv){const s2=await api('/api/upload/status/'+sv.id);
  if(s2.got){st={id:sv.id};CH=sv.ch;s2.got.forEach(([o,n])=>{have.add(o);it.done+=n})}}}catch(e){}
 if(!st){st=await api('/api/upload/start',{method:'POST',body:new URLSearchParams({size:f.size,orig:it.orig,series:it.series||''})});
  if(st.error){it.state='error';it.err=st.error;return}
  try{localStorage.setItem(fp,JSON.stringify({id:st.id,ch:CH}))}catch(e){}}
 const offs=[];for(let o=0;o<f.size;o+=CH)if(!have.has(o))offs.push(o);let next=0,failed=false;
 const worker=async()=>{while(!failed){const i=next++;if(i>=offs.length)return;const o=offs[i];let ok=false,last=0;
  for(let t=0;t<8&&!ok;t++){try{await xhr(`/api/upload/chunk/${st.id}?offset=${o}`,f.slice(o,o+CH),l=>{sent+=l-last;it.done+=l-last;last=l});ok=true}
   catch(e){sent-=last;it.done-=last;last=0;await sleep(700*(t+1))}}
  if(!ok){failed=true}}};
 await Promise.all(Array.from({length:N},worker));
 if(failed){it.state='error';it.err='Connection lost. Add the file again to resume.';return}
 const r=await api('/api/upload/finish/'+st.id,{method:'POST'});try{localStorage.removeItem(fp)}catch(e){}
 if(r.ok){it.state='done'}else{it.state='error';it.err=r.error||'Failed'}
 }catch(e){it.state='error';it.err='Upload failed'}}

/* ---- speed test ---- */
$('#run').onclick=async()=>{const b=$('#run');b.disabled=true;$('#eta5').style.display='none';
 const R={},N=+$('#streams').value||3;try{
 $('#st').textContent='Testing download...';let t=performance.now();
 const ds=await Promise.all(Array.from({length:N},(_,i)=>fetch('/api/speedtest/down?mb=32&_='+Date.now()+i).then(r=>r.arrayBuffer())));
 R.down=ds.reduce((a,x)=>a+x.byteLength,0)/((performance.now()-t)/1000)/1e6;
 $('#st').textContent='Testing upload...';const blob=new Blob([new Uint8Array(16*1048576)]);t=performance.now();
 await Promise.all(Array.from({length:N},()=>fetch('/api/speedtest/up',{method:'POST',headers:{'X-CSRF':CSRF},body:blob})));
 R.up=blob.size*N/((performance.now()-t)/1000)/1e6;
 $('#st').textContent='Testing server disk...';R.disk=(await api('/api/speedtest/disk',{method:'POST'})).mbps;
 $('#st').textContent='Done.';
 const card=(l,v)=>`<div class=card><span class=lbl>${l}</span><div class=big>${v.toFixed(2)} <small>MB/s</small></div><span class=lbl>${(v*8).toFixed(0)} Mbps</span></div>`;
 $('#res').innerHTML=card('Upload to server',R.up)+card('Download from server',R.down)+card('Server disk write',R.disk);
 const lim=Math.min(R.up,R.disk),why=R.disk<R.up*1.3?'Server storage is the limit':'Wi-Fi is the limit';
 $('#tbl').innerHTML='<tr><th>File size<th>Time<th></tr>'+[1,5,10,20].map(g=>`<tr><td>${g} GB<td>${T(g*1000/lim)}<td></tr>`).join('')+`<tr><td colspan=3 class=note>${why}. Real uploads also lose a little to chunk overhead.</tr>`;
 $('#eta5').style.display='block'}catch(e){$('#st').textContent='Test failed. Check the connection.'}
 b.disabled=false};
load();
</script></html>""".replace("__CSS__", CSS)

# ---------- run ----------
def lan_ip():
    for target in ("8.8.8.8", "10.255.255.255"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 1)); ip = s.getsockname()[0]; s.close()
            if not ip.startswith("127."): return ip
        except OSError: pass
    for cmd, rx in ((["ifconfig"], r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)"), (["ip", "-4", "addr"], r"inet (\d+\.\d+\.\d+\.\d+)")):
        try:                                   # Termux often blocks the socket trick above
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
            for ip in re.findall(rx, out):
                if not ip.startswith("127."): return ip
        except Exception: pass
    return "127.0.0.1"

def main():
    init_db()
    threading.Thread(target=sweeper, daemon=True).start()
    try:
        from waitress import serve
    except ImportError:
        raise SystemExit("Waitress is required for production use:  pip install waitress")
    log.info("Photogram on http://%s:%s  (data: %s)", lan_ip(), PORT, DATA)
    log.info("Invite code for new accounts: %s", INVITE)
    log.info("Admin import folder: %s (mode: %s)", IMPORT_DIR, IMPORT_MODE)
    log.info("ffmpeg: %s", "found (video thumbnails, MKV/MOV playback)" if FFMPEG else "not found (optional)")
    serve(app, host=HOST, port=PORT, threads=16, connection_limit=200,
          max_request_body_size=REQ_LIMIT + 1024 * 1024, channel_timeout=300, ident="photogram")

if __name__ == "__main__":
    main()
