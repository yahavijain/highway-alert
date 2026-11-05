import io
import os
import tempfile

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(
    tempfile.mkdtemp(), "test.db"
).replace("\\", "/")

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from database import Base, engine

client = TestClient(main.app)


class FakeConf:
    def __init__(self, vals):
        self.vals = vals

    def max(self):
        return max(self.vals)


class FakeBoxes(list):
    def __init__(self, vals):
        super().__init__(vals)
        self.conf = FakeConf(vals)


class FakeResult:
    def __init__(self, vals):
        self.boxes = FakeBoxes(vals)


def use_model(monkeypatch, vals):
    monkeypatch.setattr(
        main, "get_model", lambda: (lambda img, verbose=False: [FakeResult(vals)])
    )


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "PNG")
    return buf.getvalue()


def send_report(lat=28.81, lon=77.2):
    return client.post(
        "/report",
        files={"file": ("a.png", png_bytes(), "image/png")},
        data={"lat": lat, "lon": lon},
    )


def set_location(lat, heading, lon=77.2, vid="car1"):
    return client.post(
        "/location",
        data={"vehicle_id": vid, "lat": lat, "lon": lon, "heading": heading},
    )


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_haversine_known_distance():
    assert main.haversine(28.78, 77.2, 28.81, 77.2) == pytest.approx(3.34, abs=0.05)


def test_low_confidence_rejected(monkeypatch):
    use_model(monkeypatch, [0.2])
    assert send_report().json()["verified"] is False
    assert client.get("/accidents").json() == []


def test_no_detection_rejected(monkeypatch):
    use_model(monkeypatch, [])
    assert send_report().json()["verified"] is False


def test_verified_report_creates_accident(monkeypatch):
    use_model(monkeypatch, [0.9])
    r = send_report().json()
    assert r["verified"] is True
    assert len(client.get("/accidents").json()) == 1


def test_duplicate_reports_merge(monkeypatch):
    use_model(monkeypatch, [0.8])
    send_report()
    send_report()
    accidents = client.get("/accidents").json()
    assert len(accidents) == 1
    assert accidents[0]["reports"] == 2


def test_far_reports_are_separate_accidents(monkeypatch):
    use_model(monkeypatch, [0.8])
    send_report(lat=28.81)
    send_report(lat=28.90)
    assert len(client.get("/accidents").json()) == 2


def test_alert_when_approaching(monkeypatch):
    use_model(monkeypatch, [0.9])
    send_report()
    set_location(28.78, heading=0)
    alerts = client.get("/alerts/car1").json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["distance_km"] == pytest.approx(3.34, abs=0.05)


def test_no_alert_when_driving_away(monkeypatch):
    use_model(monkeypatch, [0.9])
    send_report()
    set_location(28.78, heading=180)
    assert client.get("/alerts/car1").json()["alerts"] == []


def test_no_alert_outside_radius(monkeypatch):
    use_model(monkeypatch, [0.9])
    send_report()
    set_location(28.60, heading=0)
    assert client.get("/alerts/car1").json()["alerts"] == []


def test_non_image_rejected(monkeypatch):
    use_model(monkeypatch, [0.9])
    r = client.post(
        "/report",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"lat": 28.81, "lon": 77.2},
    )
    assert r.status_code == 400


def test_close_accident_stops_alerts(monkeypatch):
    use_model(monkeypatch, [0.9])
    send_report()
    set_location(28.78, heading=0)
    acc_id = client.get("/accidents").json()[0]["id"]
    assert client.post(f"/accidents/{acc_id}/close").status_code == 200
    assert client.get("/alerts/car1").json()["alerts"] == []


def test_close_missing_accident_404():
    assert client.post("/accidents/999/close").status_code == 404