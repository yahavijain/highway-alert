import io, math
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Highway Accident Alert")
model = YOLO("best.pt")

CONF_THRESHOLD = 0.5
ALERT_RADIUS_KM = 5
vehicles = {}    # vehicle_id -> {lat, lon, heading}
accidents = []   # verified accidents

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def moving_toward(v, acc):
    b = bearing(v["lat"], v["lon"], acc["lat"], acc["lon"])
    return abs((v["heading"] - b + 180) % 360 - 180) <= 60

@app.post("/location")
def update_location(vehicle_id: str = Form(...), lat: float = Form(...),
                    lon: float = Form(...), heading: float = Form(...)):
    vehicles[vehicle_id] = {"lat": lat, "lon": lon, "heading": heading}
    return {"ok": True}

@app.post("/report")
async def report(file: UploadFile = File(...), lat: float = Form(...),
                 lon: float = Form(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    boxes = model(img, verbose=False)[0].boxes
    conf = float(boxes.conf.max()) if len(boxes) else 0.0
    if conf < CONF_THRESHOLD:
        return {"verified": False, "confidence": conf}
    acc = {"id": len(accidents) + 1, "lat": lat, "lon": lon, "confidence": conf}
    accidents.append(acc)
    return {"verified": True, "accident": acc}

@app.get("/alerts/{vehicle_id}")
def alerts(vehicle_id: str):
    v = vehicles.get(vehicle_id)
    if not v:
        return {"alerts": []}
    out = []
    for a in accidents:
        d = haversine(v["lat"], v["lon"], a["lat"], a["lon"])
        if d <= ALERT_RADIUS_KM and moving_toward(v, a):
            out.append({"accident_id": a["id"], "distance_km": round(d, 2),
                        "message": f"Accident {d:.1f} km ahead. Slow down."})
    return {"alerts": out}