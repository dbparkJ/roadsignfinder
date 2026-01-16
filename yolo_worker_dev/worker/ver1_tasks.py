import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from minio import Minio
from ultralytics import YOLO
from .celery_app import celery

# ✅ DB (sync)
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://jm:jm1234@111.111.111.216:5432/jm")
if not DB_URL:
    raise RuntimeError("DB_URL env var is missing. e.g. export DB_URL='postgresql+psycopg2://jm:jm1234@111.111.111.216:5432/jm'")

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ✅ MinIO
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "geonws")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "geonws1234")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "photos")

minio = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=MINIO_SECURE)

# --- YOLO 모델 (WARM: 프로세스에서 1번만 로딩)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/geonws/roadsign_finder/yolo_worker/model/seongbin_model.pt")
_yolo = None
def get_model():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(MODEL_PATH)
    return _yolo

@celery.task(name="run_inference", bind=True)
def run_inference(self, job_id: str):
    tmp_path = None
    db = SessionLocal()
    try:
        # (중요) 중복 처리 방지: queued인 job만 running으로 "선점"
        claimed = db.execute(text("""
          UPDATE inference_jobs
          SET status='running', started_at=now(), progress=0.1
          WHERE id=:job_id AND status='queued'
          RETURNING photo_id, member_id
        """), {"job_id": job_id}).mappings().first()
        db.commit()
        if not claimed:
            return  # 이미 다른 워커가 처리 중/완료

        photo = db.execute(text("""
          SELECT object_key
          FROM photos
          WHERE id=:photo_id
        """), {"photo_id": claimed["photo_id"]}).mappings().first()
        if not photo:
            raise RuntimeError("photo not found")

        object_key = photo["object_key"]

        # MinIO 다운로드
        tmp_path = f"/tmp/{job_id}.jpg"
        minio.fget_object(MINIO_BUCKET, object_key, tmp_path)
        db.execute(text("UPDATE inference_jobs SET progress=0.3 WHERE id=:job_id"), {"job_id": job_id})
        db.commit()

        # YOLO Seg 추론 (GPU는 환경/쿠버에서 할당)
        model = get_model()
        results = model.predict(tmp_path, device=0)
        r0 = results[0]

        # 기존 결과 삭제
        db.execute(text("DELETE FROM segmentations WHERE job_id=:job_id"), {"job_id": job_id})

        if r0.masks is not None and len(r0.masks) > 0:
            polys = r0.masks.xy   # 객체별 폴리곤(픽셀좌표)
            names = r0.names

            for i, seg in enumerate(polys):
                cls = int(r0.boxes.cls[i].item())
                score = float(r0.boxes.conf[i].item())

                # seg: [[x,y], ...] -> WKT Polygon
                pts = [(float(x), float(y)) for x, y in seg]
                if len(pts) < 3:
                    continue
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                wkt = "POLYGON((" + ",".join([f"{x} {y}" for x, y in pts]) + "))"

                db.execute(text("""
                INSERT INTO segmentations(job_id, photo_id, class_id, class_name, score, geom_px)
                VALUES(:job_id, :photo_id, :class_id, :class_name, :score,
                ST_Multi(ST_GeomFromText(:wkt)))
                """), {
                "job_id": job_id,
                "photo_id": claimed["photo_id"],
                "class_id": cls,
                "class_name": names.get(cls),
                "score": score,
                "wkt": wkt,
                })


        db.execute(text("""
          UPDATE inference_jobs
          SET status='succeeded', progress=1, finished_at=now()
          WHERE id=:job_id
        """), {"job_id": job_id})
        db.commit()

    except Exception as e:
        db.rollback()  # ✅ 이거 없으면 다음 SQL은 전부 InFailedSqlTransaction 터짐

        db.execute(text("""
        UPDATE inference_jobs
        SET status='failed', error=:err, finished_at=now()
        WHERE id=:job_id
        """), {"job_id": job_id, "err": str(e)})

        db.commit()
        raise

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        db.close()

