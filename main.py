import io, math, os, uuid, logging
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Vehicle, Accident, Report, utcnow

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("highway-alert")

Base.metadata.create_all(engine)
app = FastAPI(title="Highway Accident Alert", version="1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model

CONF_THRESHOLD = 0.5
ALERT_RADIUS_KM = 5
MERGE_RADIUS_KM = 0.3
VEHICLE_FRESH = timedelta(minutes=5)
ACCIDENT_TTL = timedelta(hours=3)
MAX_UPLOAD = 5 * 1024 * 1024
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def moving_toward(v, a):
    b = bearing(v.lat, v.lon, a.lat, a.lon)
    return abs((v.heading - b + 180) % 360 - 180) <= 60


def active_accidents(db: Session):
    cutoff = utcnow() - ACCIDENT_TTL
    return db.scalars(
        select(Accident).where(Accident.status == "active", Accident.created_at >= cutoff)
    ).all()


def nearby_active_accident(db: Session, lat, lon):
    for a in active_accidents(db):
        if haversine(lat, lon, a.lat, a.lon) <= MERGE_RADIUS_KM:
            return a
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/location")
def update_location(
    vehicle_id: str = Form(..., max_length=64),
    lat: float = Form(..., ge=-90, le=90),
    lon: float = Form(..., ge=-180, le=180),
    heading: float = Form(0, ge=0, le=360),
    db: Session = Depends(get_db),
):
    v = db.get(Vehicle, vehicle_id)
    if v:
        v.lat, v.lon, v.heading = lat, lon, heading
        v.updated_at = utcnow()
    else:
        db.add(Vehicle(id=vehicle_id, lat=lat, lon=lon, heading=heading))
    db.commit()
    return {"ok": True}


@app.post("/report")
def report(
    file: UploadFile = File(...),
    lat: float = Form(..., ge=-90, le=90),
    lon: float = Form(..., ge=-180, le=180),
    db: Session = Depends(get_db),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 5 MB)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    boxes = get_model()(img, verbose=False)[0].boxes
    conf = float(boxes.conf.max()) if len(boxes) else 0.0
    if conf < CONF_THRESHOLD:
        log.info("Report rejected (confidence %.2f)", conf)
        return {"verified": False, "confidence": round(conf, 3)}

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
    img.save(path, "JPEG")

    acc = nearby_active_accident(db, lat, lon)
    if acc:
        acc.confidence = max(acc.confidence, conf)
    else:
        acc = Accident(lat=lat, lon=lon, confidence=conf)
        db.add(acc)
        db.flush()
    db.add(Report(accident_id=acc.id, confidence=conf, image_path=str(path)))
    db.commit()
    log.info("Accident %s verified (confidence %.2f)", acc.id, conf)
    return {
        "verified": True,
        "accident": {
            "id": acc.id, "lat": acc.lat, "lon": acc.lon,
            "confidence": round(acc.confidence, 3), "reports": len(acc.reports),
        },
    }


@app.get("/alerts/{vehicle_id}")
def alerts(vehicle_id: str, db: Session = Depends(get_db)):
    v = db.get(Vehicle, vehicle_id)
    if not v or utcnow() - v.updated_at > VEHICLE_FRESH:
        return {"alerts": []}
    out = []
    for a in active_accidents(db):
        d = haversine(v.lat, v.lon, a.lat, a.lon)
        if d <= ALERT_RADIUS_KM and moving_toward(v, a):
            out.append({
                "accident_id": a.id,
                "distance_km": round(d, 2),
                "message": f"Accident {d:.1f} km ahead. Slow down.",
            })
    return {"alerts": out}


@app.get("/accidents")
def list_accidents(db: Session = Depends(get_db)):
    return [
        {"id": a.id, "lat": a.lat, "lon": a.lon,
         "confidence": round(a.confidence, 3), "reports": len(a.reports)}
        for a in active_accidents(db)
    ]


@app.post("/accidents/{accident_id}/close")
def close_accident(accident_id: int, db: Session = Depends(get_db)):
    a = db.get(Accident, accident_id)
    if not a:
        raise HTTPException(404, "Accident not found")
    a.status = "closed"
    db.commit()
    return {"closed": accident_id}