import os, asyncio, datetime as dt
from typing import Optional
import requests, sqlalchemy as sa, cloudinary, cloudinary.utils
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

cloudinary.config(secure=True)  # reads CLOUDINARY_URL
DB = os.getenv("DATABASE_URL", "sqlite:///./tracex.db").replace("postgres://", "postgresql://", 1)
KEY = os.getenv("WORKER_KEY", "dev-key")
engine = sa.create_engine(DB, pool_pre_ping=True)
md = sa.MetaData()
S = sa.String
cameras = sa.Table("cameras", md, sa.Column("id", S(40), primary_key=True), sa.Column("name", S(100)),
                   sa.Column("lat", sa.Float), sa.Column("lng", sa.Float))
sightings = sa.Table("sightings", md, sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                     sa.Column("plate", S(20), index=True), sa.Column("camera_id", S(40)),
                     sa.Column("captured_at", sa.DateTime, index=True), sa.Column("confidence", sa.Float),
                     sa.Column("vehicle_pid", S(300)), sa.Column("plate_pid", S(300)))
blacklist = sa.Table("blacklist", md, sa.Column("plate", S(20), primary_key=True), sa.Column("reason", S(200)))
alerts = sa.Table("alerts", md, sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                  sa.Column("plate", S(20)), sa.Column("camera_id", S(40)),
                  sa.Column("captured_at", sa.DateTime), sa.Column("reason", S(200)))
md.create_all(engine)
app = FastAPI(title="TraceX")
sockets: list = []

def need_key(k):
    if k != KEY: raise HTTPException(401, "Bad API key")

def signed(pid, kind):
    if not pid: return None
    t = ([{"effect": "sharpen"}, {"width": 360, "crop": "limit"}] if kind == "plate" else
         [{"effect": "blur_faces:600"}, {"width": 520, "crop": "limit"}]) + [{"quality": "auto", "fetch_format": "auto"}]
    try:
        return cloudinary.utils.cloudinary_url(pid, type="authenticated", sign_url=True, transformation=t)[0]
    except Exception:
        return None

def row(r, cams):
    d = dict(r._mapping); c = cams.get(d["camera_id"], {})
    d["captured_at"] = d["captured_at"].isoformat() + "Z"
    d.update(camera=c.get("name", d["camera_id"]), lat=c.get("lat"), lng=c.get("lng"),
             vehicle_url=signed(d.pop("vehicle_pid"), "vehicle"), plate_url=signed(d.pop("plate_pid"), "plate"))
    return d

def cam_map(cx):
    return {c.id: dict(name=c.name, lat=c.lat, lng=c.lng) for c in cx.execute(sa.select(cameras))}

class Cam(BaseModel):
    id: str; name: str; lat: float; lng: float

class Sighting(BaseModel):
    plate: str; camera_id: str; confidence: float = 0
    captured_at: Optional[dt.datetime] = None
    vehicle_pid: Optional[str] = None; plate_pid: Optional[str] = None

class Black(BaseModel):
    plate: str; reason: str = "Flagged"

def norm(p): return "".join(ch for ch in p.upper() if ch.isalnum())

async def broadcast(msg):
    for w in list(sockets):
        try: await w.send_json(msg)
        except Exception: sockets.remove(w)

def telegram(text):
    t, c = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if t and c:
        try: requests.post(f"https://api.telegram.org/bot{t}/sendMessage", json={"chat_id": c, "text": text}, timeout=5)
        except Exception: pass

@app.post("/api/cameras")
def add_camera(c: Cam, x_api_key: str = Header("")):
    need_key(x_api_key)
    with engine.begin() as cx:
        cx.execute(cameras.delete().where(cameras.c.id == c.id)); cx.execute(cameras.insert().values(**c.model_dump()))
    return {"ok": True}

@app.get("/api/cameras")
def list_cameras():
    with engine.connect() as cx:
        return [dict(r._mapping) for r in cx.execute(sa.select(cameras))]

@app.post("/api/sightings")
async def add_sighting(s: Sighting, x_api_key: str = Header("")):
    need_key(x_api_key)
    plate = norm(s.plate)
    when = (s.captured_at or dt.datetime.now(dt.timezone.utc))
    when = (when.astimezone(dt.timezone.utc) if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)).replace(tzinfo=None)
    with engine.begin() as cx:
        dup = cx.execute(sa.select(sightings.c.id).where(sightings.c.plate == plate, sightings.c.camera_id == s.camera_id,
                         sightings.c.captured_at > when - dt.timedelta(seconds=10))).first()
        if dup: return {"duplicate": True}
        sid = cx.execute(sightings.insert().values(plate=plate, camera_id=s.camera_id, captured_at=when,
              confidence=s.confidence, vehicle_pid=s.vehicle_pid, plate_pid=s.plate_pid)).inserted_primary_key[0]
        b = cx.execute(sa.select(blacklist).where(blacklist.c.plate == plate)).first()
        if b:
            cx.execute(alerts.insert().values(plate=plate, camera_id=s.camera_id, captured_at=when, reason=b.reason))
        name = cam_map(cx).get(s.camera_id, {}).get("name", s.camera_id)
    if b:
        msg = {"plate": plate, "camera": name, "captured_at": when.isoformat() + "Z", "reason": b.reason}
        await broadcast(msg); telegram(f"ALERT {plate} seen at {name} ({b.reason})")
    return {"id": sid, "alert": bool(b)}

@app.get("/api/vehicles/{plate}/trajectory")
def trajectory(plate: str):
    plate = norm(plate)
    with engine.connect() as cx:
        cams = cam_map(cx)
        rows = cx.execute(sa.select(sightings).where(sightings.c.plate == plate).order_by(sightings.c.captured_at)).fetchall()
        b = cx.execute(sa.select(blacklist).where(blacklist.c.plate == plate)).first()
    return {"plate": plate, "blacklisted": bool(b), "reason": b.reason if b else None, "points": [row(r, cams) for r in rows]}

@app.get("/api/sightings")
def recent(limit: int = 20):
    with engine.connect() as cx:
        cams = cam_map(cx)
        return [row(r, cams) for r in cx.execute(sa.select(sightings).order_by(sightings.c.captured_at.desc()).limit(limit))]

@app.get("/api/alerts")
def recent_alerts():
    with engine.connect() as cx:
        cams = cam_map(cx)
        return [dict(plate=a.plate, camera=cams.get(a.camera_id, {}).get("name", a.camera_id), reason=a.reason,
                     captured_at=a.captured_at.isoformat() + "Z")
                for a in cx.execute(sa.select(alerts).order_by(alerts.c.captured_at.desc()).limit(20))]

@app.get("/api/blacklist")
def get_black():
    with engine.connect() as cx:
        return [dict(r._mapping) for r in cx.execute(sa.select(blacklist))]

@app.post("/api/blacklist")
def add_black(b: Black):
    p = norm(b.plate)
    with engine.begin() as cx:
        cx.execute(blacklist.delete().where(blacklist.c.plate == p)); cx.execute(blacklist.insert().values(plate=p, reason=b.reason))
    return {"ok": True}

@app.delete("/api/blacklist/{plate}")
def del_black(plate: str):
    with engine.begin() as cx: cx.execute(blacklist.delete().where(blacklist.c.plate == norm(plate)))
    return {"ok": True}

@app.post("/api/demo/seed")
async def seed():
    """Loads sample cameras and a blacklisted vehicle's route so the dashboard works without a worker."""
    cams = [("cam01", "MG Road Junction", 12.9756, 77.6066), ("cam02", "Trinity Circle", 12.9730, 77.6200),
            ("cam03", "Indiranagar 100ft Rd", 12.9784, 77.6408), ("cam04", "Domlur Flyover", 12.9609, 77.6387),
            ("cam05", "Silk Board Junction", 12.9177, 77.6233)]
    with engine.begin() as cx:
        for i, n, la, ln in cams:
            cx.execute(cameras.delete().where(cameras.c.id == i)); cx.execute(cameras.insert().values(id=i, name=n, lat=la, lng=ln))
        cx.execute(blacklist.delete().where(blacklist.c.plate == "KA01AB1234"))
        cx.execute(blacklist.insert().values(plate="KA01AB1234", reason="Reported stolen"))
    t0 = dt.datetime.utcnow() - dt.timedelta(minutes=40)
    for k, (i, *_ ) in enumerate(cams):
        await add_sighting(Sighting(plate="KA01AB1234", camera_id=i, confidence=0.9 - k * 0.02,
                           captured_at=(t0 + dt.timedelta(minutes=k * 9)).replace(tzinfo=dt.timezone.utc)), KEY)
    for i in ("cam01", "cam03"):
        await add_sighting(Sighting(plate="KA05MN4821", camera_id=i, confidence=0.87,
                           captured_at=(t0 + dt.timedelta(minutes=12)).replace(tzinfo=dt.timezone.utc)), KEY)
    return {"ok": True}

@app.websocket("/ws")
async def ws(w: WebSocket):
    await w.accept(); sockets.append(w)
    try:
        while True: await w.receive_text()
    except WebSocketDisconnect:
        if w in sockets: sockets.remove(w)

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
