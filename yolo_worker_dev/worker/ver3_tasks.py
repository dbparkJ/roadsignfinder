# yolo_worker/tasks.py
import os
import gc
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from minio import Minio
from ultralytics import YOLO
from PIL import Image
import torch

from .celery_app import celery
from .crop_store import ensure_bucket, cleanup_prefix, save_crop_and_upload
from .pred_store import save_pred_and_upload


# ✅ DB (sync)
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://jm:jm1234@111.111.111.216:5432/jm")
if not DB_URL:
    raise RuntimeError("DB_URL env var is missing.")

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# ✅ MinIO
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "111.111.111.216:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "geonws")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "geonws1234")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "photos")

# ✅ Crop / Pred 버킷
MINIO_CROP_BUCKET = os.getenv("MINIO_CROP_BUCKET", "crop")
MINIO_PRED_BUCKET = os.getenv("MINIO_PRED_BUCKET", "pred")

minio = Minio(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=MINIO_SECURE)

# 버킷 준비(없으면 생성)
ensure_bucket(minio, MINIO_BUCKET)
ensure_bucket(minio, MINIO_CROP_BUCKET)
ensure_bucket(minio, MINIO_PRED_BUCKET)

# --- YOLO 모델 (WARM)
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/home/geonws/roadsign_finder/yolo_worker/model/seungjune.pt")
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
          RETURNING photo_id, member_id
        """), {"job_id": job_id}).mappings().first()
        db.commit()
        if not claimed:
            return

        # 2) 원본 object_key 조회
        photo = db.execute(text("""
          SELECT object_key
          FROM photos
          WHERE id=:photo_id
        """), {"photo_id": claimed["photo_id"]}).mappings().first()
        if not photo:
            raise RuntimeError("photo not found")
        object_key = photo["object_key"]

        # 3) 원본 다운로드 (확장자 유지)
        ext = os.path.splitext(object_key)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            ext = ".jpg"

        tmp_path = f"/tmp/{job_id}{ext}"
        minio.fget_object(MINIO_BUCKET, object_key, tmp_path)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"download failed or empty: {tmp_path}")

        db.execute(text("UPDATE inference_jobs SET progress=30 WHERE id=:job_id"), {"job_id": job_id})
        db.commit()

        # 4) 추론 (메모리 절감 옵션)
        model = get_model()
        results = model.predict(
            tmp_path,
            device=0,
            imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
            batch=1,
            half=os.getenv("YOLO_HALF", "true").lower() == "true",
            retina_masks=os.getenv("YOLO_RETINA_MASKS", "false").lower() == "true",
            verbose=False,
        )

        # ✅ GPU 결과를 CPU로 내리고 GPU 참조 제거
        r0 = results[0].cpu()
        del results
        _cleanup_cuda()

        db.execute(text("UPDATE inference_jobs SET progress=60 WHERE id=:job_id"), {"job_id": job_id})
        db.commit()

        # 5) 기존 결과 정리(DB)
        db.execute(text("DELETE FROM segmentations WHERE job_id=:job_id"), {"job_id": job_id})
        db.execute(text("DELETE FROM crops WHERE job_id=:job_id"), {"job_id": job_id})
        db.execute(text("DELETE FROM inference_images WHERE job_id=:job_id"), {"job_id": job_id})

        # MinIO 결과도 재실행 대비 정리(원하면 env로 끌 수도 있음)
        cleanup_prefix(minio, MINIO_CROP_BUCKET, prefix=f"{job_id}/")
        cleanup_prefix(minio, MINIO_PRED_BUCKET, prefix=f"{job_id}/")

        # 원본 이미지 로드(크롭용)
        img = Image.open(tmp_path).convert("RGB")

        # ✅ 탐지 결과 존재 여부 판단(박스 or 마스크 중 하나라도 있으면 True)
        boxes = r0.boxes
        masks = r0.masks
        has_boxes = (boxes is not None and len(boxes) > 0)
        has_masks = (masks is not None and len(masks) > 0)
        has_det = has_boxes or has_masks

        # 6) ✅ (조건) 탐지 결과가 있을 때만 pred(오버레이 이미지) 저장
        if has_det:
            pred_object_key = f"{job_id}/pred.jpg"
            pred_object_key = save_pred_and_upload(
                minio=minio,
                result=r0,  # CPU 결과 (plot도 GPU 추가 점유 거의 없음)
                pred_bucket=MINIO_PRED_BUCKET,
                object_key=pred_object_key,
            )

            db.execute(text("""
                INSERT INTO inference_images(job_id, photo_id, bucket, object_key)
                VALUES(:job_id, :photo_id, :bucket, :object_key)
            """), {
                "job_id": job_id,
                "photo_id": claimed["photo_id"],
                "bucket": MINIO_PRED_BUCKET,
                "object_key": pred_object_key,
            })

        # 7) 세그 + crop 저장 (박스가 있어야 의미 있음)
        names = r0.names
        if has_boxes:
            xyxy = boxes.xyxy
            clses = boxes.cls
            confs = boxes.conf

            for i in range(len(boxes)):
                cls = int(clses[i].item())
                score = float(confs[i].item())
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
                class_name = names.get(cls)

                # (A) 마스크가 있으면 폴리곤 저장
                if has_masks and len(r0.masks.xy) > i:
                    seg = r0.masks.xy[i]
                    pts = [(float(x), float(y)) for x, y in seg]
                    if len(pts) >= 3:
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
                            "class_name": class_name,
                            "score": score,
                            "wkt": wkt,
                        })

                # (B) bbox crop 저장 → MinIO 업로드 → crops 테이블 저장
                crop_object_key = f"{job_id}/{i}_{cls}_{int(score*1000)}.jpg"
                crop_object_key, (x1i, y1i, x2i, y2i) = save_crop_and_upload(
                    minio=minio,
                    src_img=img,
                    crop_bucket=MINIO_CROP_BUCKET,
                    object_key=crop_object_key,
                    bbox_xyxy=(x1, y1, x2, y2),
                )

                db.execute(text("""
                    INSERT INTO crops(
                        job_id, photo_id,
                        class_id, class_name, score,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        bucket, object_key
                    )
                    VALUES(
                        :job_id, :photo_id,
                        :class_id, :class_name, :score,
                        :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                        :bucket, :object_key
                    )
                """), {
                    "job_id": job_id,
                    "photo_id": claimed["photo_id"],
                    "class_id": cls,
                    "class_name": class_name,
                    "score": score,
                    "bbox_x1": float(x1i),
                    "bbox_y1": float(y1i),
                    "bbox_x2": float(x2i),
                    "bbox_y2": float(y2i),
                    "bucket": MINIO_CROP_BUCKET,
                    "object_key": crop_object_key,
                })

        db.execute(text("""
          UPDATE inference_jobs
          SET status='succeeded', progress=100, finished_at=now()
          WHERE id=:job_id
        """), {"job_id": job_id})
        db.commit()

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

