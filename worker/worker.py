"""TraceX AI worker: vehicle detect -> ByteTrack -> plate crop -> OCR -> vote -> Cloudinary upload -> API."""
import os, io, re, argparse, datetime as dt
from collections import Counter
import cv2, requests, cloudinary, cloudinary.uploader
from ultralytics import YOLO

PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")  # Indian format e.g. KA01AB1234
p = argparse.ArgumentParser()
p.add_argument("--source", required=True, help="video file, RTSP url, or 0 for webcam")
p.add_argument("--camera-id", required=True); p.add_argument("--name", default=None)
p.add_argument("--lat", type=float, required=True); p.add_argument("--lng", type=float, required=True)
p.add_argument("--api", default=os.getenv("API_URL", "http://localhost:8000"))
a = p.parse_args()
KEY = os.getenv("WORKER_KEY", "dev-key"); H = {"x-api-key": KEY}
STRIDE, LOST, MIN_VOTES = 3, 60, 3
cloudinary.config(secure=True)  # reads CLOUDINARY_URL

vehicle_model = YOLO(os.getenv("VEHICLE_MODEL", "yolo11n.pt"))
PM = os.getenv("PLATE_MODEL", "plate.pt")  # your fine-tuned plate detector (optional)
plate_model = YOLO(PM) if os.path.exists(PM) else None

if os.getenv("OCR_ENGINE") == "paddle":
    from paddleocr import PaddleOCR
    _p = PaddleOCR(use_angle_cls=True, lang="en")
    def ocr(img): return [(t, c) for pg in (_p.ocr(img, cls=True) or []) for _, (t, c) in (pg or [])]
else:
    import easyocr
    _r = easyocr.Reader(["en"], gpu=False)
    def ocr(img): return [(t, c) for _, t, c in _r.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")]

def find_plate(vcrop):
    if plate_model:
        b = plate_model(vcrop, verbose=False)[0].boxes
        if len(b) == 0: return None
        x1, y1, x2, y2 = b.xyxy[b.conf.argmax()].int().tolist()
        return vcrop[y1:y2, x1:x2]
    return vcrop[int(vcrop.shape[0] * 0.5):, :]  # fallback: lower half of the vehicle

def read_plate(pc):
    if pc is None or pc.size == 0: return None
    g = cv2.resize(cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res = ocr(g)
    if not res: return None
    txt = re.sub(r"[^A-Z0-9]", "", "".join(t for t, _ in res).upper())
    return (txt, sum(c for _, c in res) / len(res)) if PLATE_RE.match(txt) else None

def upload(img, folder, plate, ts):
    ok, buf = cv2.imencode(".jpg", img)
    r = cloudinary.uploader.upload(io.BytesIO(buf.tobytes()), folder=f"tracex/{a.camera_id}/{folder}",
        type="authenticated", tags=["tracex", folder, a.camera_id],
        context={"plate": plate, "camera_id": a.camera_id, "captured_at": ts, "lat": str(a.lat), "lng": str(a.lng)})
    return r["public_id"]

def send(t):
    plate, n = t["votes"].most_common(1)[0]
    conf = sum(t["conf"][plate]) / len(t["conf"][plate])
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        body = dict(plate=plate, camera_id=a.camera_id, confidence=round(conf, 3), captured_at=ts,
                    vehicle_pid=upload(t["v"], "vehicles", plate, ts), plate_pid=upload(t["p"], "plates", plate, ts))
        print("sent", requests.post(f"{a.api}/api/sightings", json=body, headers=H, timeout=15).json())
    except Exception as e:
        print("send failed:", e)
    t["sent"] = True

requests.post(f"{a.api}/api/cameras", json=dict(id=a.camera_id, name=a.name or a.camera_id, lat=a.lat, lng=a.lng), headers=H, timeout=15)
tracks, idx = {}, 0
src = int(a.source) if a.source.isdigit() else a.source
for r in vehicle_model.track(source=src, persist=True, tracker="bytetrack.yaml", classes=[2, 3, 5, 7], stream=True, verbose=False):
    idx += 1
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist()):
            t = tracks.setdefault(tid, dict(votes=Counter(), conf={}, sent=False, best=0, v=None, p=None, last=idx))
            t["last"] = idx
            if t["sent"] or idx % STRIDE or x2 - x1 < 80 or y2 - y1 < 60: continue
            v = r.orig_img[max(y1, 0):y2, max(x1, 0):x2]
            pc = find_plate(v); res = read_plate(pc)
            if not res: continue
            txt, c = res
            t["votes"][txt] += 1; t["conf"].setdefault(txt, []).append(c)
            sharp = cv2.Laplacian(cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp > t["best"]: t.update(best=sharp, v=v.copy(), p=pc.copy())
            if t["votes"][txt] >= MIN_VOTES and not t["sent"]: send(t)
    for tid in [k for k, t in tracks.items() if idx - t["last"] > LOST]:
        t = tracks.pop(tid)
        if t["votes"] and not t["sent"] and t["votes"].most_common(1)[0][1] >= 2: send(t)
for t in tracks.values():
    if t["votes"] and not t["sent"] and t["votes"].most_common(1)[0][1] >= 2: send(t)
