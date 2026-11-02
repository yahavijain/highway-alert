from datetime import datetime, timezone
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class Vehicle(Base):
    __tablename__ = "vehicles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    lat: Mapped[float]
    lon: Mapped[float]
    heading: Mapped[float]
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

class Accident(Base):
    __tablename__ = "accidents"
    id: Mapped[int] = mapped_column(primary_key=True)
    lat: Mapped[float]
    lon: Mapped[float]
    confidence: Mapped[float]
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    reports: Mapped[list["Report"]] = relationship(back_populates="accident")

class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    accident_id: Mapped[int] = mapped_column(ForeignKey("accidents.id"), index=True)
    confidence: Mapped[float]
    image_path: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    accident: Mapped["Accident"] = relationship(back_populates="reports")