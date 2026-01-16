# yolo_worker/tasks.py
import os
import gc
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from minio import Minio
from ultralytics import YOLO
import torch

from .celery_app import celery
from .crop_store import ensure_bucket
from .postprocess_from_db import postprocess_from_db

# ✅ DB (sync)
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://jm:jm1234@111.111.111.216:5432/jm")
engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ✅ MinIO
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "geonws")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "geonws1234")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

MINIO_BUCKET      = os.getenv("MINIO_BUCKET", "photos")
MINIO_CROP_BUCKET = os.getenv("MINIO_CROP_BUCKET", "crop")
MINIO_PRED_BUCKET = os.getenv("MINIO_PRED_BUCKET", "pred")

minio = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=MINIO_SECURE)
ensure_bucket(minio, MINIO_BUCKET)
ensure_bucket(minio, MINIO_CROP_BUCKET)
ensure_bucket(minio, MINIO_PRED_BUCKET)

# --- YOLO 모델 (WARM)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/geonws/roadsign_finder/yolo_worker/model/best.pt")
_yolo = None
def get_model():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(MODEL_PATH)
    return _yolo


@celery.task(name="run_inference", bind=True)
def run_inference(self, job_id: str):
    db = SessionLocal()
    tmp_path = None

    def _cleanup_cuda():
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        # 1) queued job 선점
        claimed = db.execute(text("""
          UPDATE inference_jobs
          SET status='running', started_at=now(), progress=10
          WHERE id=:job_id AND status='queued'
          RETURNING photo_id
        """), {"job_id": job_id}).mappings().first()
        db.commit()
        if not claimed:
            return

        photo_id = claimed["photo_id"]

        # 2) 원본 object_key 조회
        photo = db.execute(text("""
          SELECT object_key
          FROM photos
          WHERE id=:photo_id
        """), {"photo_id": photo_id}).mappings().first()
        if not photo:
            raise RuntimeError("photo not found")
        object_key = photo["object_key"]

        # 3) 원본 다운로드
        ext = os.path.splitext(object_key)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            ext = ".jpg"
        tmp_path = f"/tmp/{job_id}{ext}"
        minio.fget_object(MINIO_BUCKET, object_key, tmp_path)

        db.execute(text("UPDATE inference_jobs SET progress=30 WHERE id=:job_id"), {"job_id": job_id})
        db.commit()

        # 4) 추론
        model = get_model()
        results = model.predict(
            tmp_path,
            device=0,
            #imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
            #batch=1,
            #half=os.getenv("YOLO_HALF", "true").lower() == "true",
            #retina_masks=os.getenv("YOLO_RETINA_MASKS", "false").lower() == "true",
            #verbose=False,
        )

        r0 = results[0].cpu()
        del results
        _cleanup_cuda()

        db.execute(text("UPDATE inference_jobs SET progress=55 WHERE id=:job_id"), {"job_id": job_id})
        db.commit()

        # 5) 기존 결과 정리(원본 추론결과 테이블)
        db.execute(text("DELETE FROM segmentations WHERE job_id=:job_id"), {"job_id": job_id})
        db.execute(text("DELETE FROM detections WHERE job_id=:job_id"), {"job_id": job_id})
        db.execute(text("DELETE FROM crops WHERE job_id=:job_id"), {"job_id": job_id})
        db.execute(text("DELETE FROM inference_images WHERE job_id=:job_id"), {"job_id": job_id})

        # 6) detections + segmentations 저장
        boxes = r0.boxes
        masks = r0.masks
        has_boxes = boxes is not None and len(boxes) > 0
        has_masks = masks is not None and len(masks) > 0
        names = r0.names

        if has_boxes:
            xyxy = boxes.xyxy
            clses = boxes.cls
            confs = boxes.conf

            for i in range(len(boxes)):
                cls = int(clses[i].item())
                score = float(confs[i].item())
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                class_name = names.get(cls) or str(cls)

                db.execute(text("""
                    INSERT INTO detections(job_id, photo_id, det_index, class_id, class_name, score,
                                           bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES(:job_id, :photo_id, :det_index, :class_id, :class_name, :score,
                           :x1, :y1, :x2, :y2)
                """), {
                    "job_id": job_id,
                    "photo_id": photo_id,
                    "det_index": i,
                    "class_id": cls,
                    "class_name": class_name,
                    "score": score,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

                # 마스크(있으면) segmentations 저장 (det_index 포함)
                if has_masks and len(r0.masks.xy) > i:
                    seg = r0.masks.xy[i]
                    pts = [(float(x), float(y)) for x, y in seg]
                    if len(pts) >= 3:
                        if pts[0] != pts[-1]:
                            pts.append(pts[0])
                        wkt = "POLYGON((" + ",".join([f"{x} {y}" for x, y in pts]) + "))"
                        db.execute(text("""
                            INSERT INTO segmentations(job_id, photo_id, det_index, class_id, class_name, score, geom_px)
                            VALUES(:job_id, :photo_id, :det_index, :class_id, :class_name, :score,
                                   ST_Multi(ST_GeomFromText(:wkt)))
                        """), {
                            "job_id": job_id,
                            "photo_id": photo_id,
                            "det_index": i,
                            "class_id": cls,
                            "class_name": class_name,
                            "score": score,
                            "wkt": wkt,
                        })

        db.commit()

        # ✅ 여기서부터는 “DB에 저장된 결과만” 사용해서 crop/pred 생성
        det_count = postprocess_from_db(
            db=db,
            minio=minio,
            job_id=job_id,
            photo_id=str(photo_id),
            original_img_path=tmp_path,
            crop_bucket=MINIO_CROP_BUCKET,
            pred_bucket=MINIO_PRED_BUCKET,
        )
        db.commit()

        db.execute(text("""
          UPDATE inference_jobs
          SET status='succeeded', progress=100, finished_at=now()
          WHERE id=:job_id
        """), {"job_id": job_id})
        db.commit()

        # det_count == 0 이면 postprocess_from_db가 pred 저장 안 함(요구사항 만족)

    except Exception as e:
        db.rollback()
        db.execute(text("""
            UPDATE inference_jobs
            SET status='failed', error=:err, finished_at=now()
            WHERE id=:job_id
        """), {"job_id": job_id, "err": str(e)})
        db.commit()
        raise

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        _cleanup_cuda()
        db.close()

