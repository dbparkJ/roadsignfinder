# yolo_worker/postprocess_from_db.py
import os, json, uuid
from PIL import Image
from sqlalchemy import text

from .yolo_overlay import Det, draw_yolo_style


def _clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def postprocess_from_db(
    *,
    db,
    minio,
    job_id: str,
    photo_id: str,
    original_img_path: str,
    crop_bucket: str,
    pred_bucket: str,
):
    """
    DB(detections/segmentations)에서 결과 조회 → crop/pred 생성 및 저장.
    탐지 0개면 pred 저장 안 함.
    """
    det_rows = db.execute(text("""
        SELECT det_index, class_id, COALESCE(class_name,'') AS class_name, score,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM detections
        WHERE job_id = :job_id
        ORDER BY det_index ASC
    """), {"job_id": job_id}).mappings().all()

    if not det_rows:
        return 0

    seg_rows = db.execute(text("""
        SELECT det_index, ST_AsGeoJSON(geom_px) AS geom_json
        FROM segmentations
        WHERE job_id = :job_id
    """), {"job_id": job_id}).mappings().all()

    seg_map = {}
    for r in seg_rows:
        gj = r.get("geom_json")
        if gj:
            try:
                seg_map[int(r["det_index"])] = json.loads(gj)
            except Exception:
                pass

    img = Image.open(original_img_path).convert("RGB")
    w, h = img.size

    # 기존 후처리 결과 정리(DB)
    db.execute(text("DELETE FROM crops WHERE job_id=:job_id"), {"job_id": job_id})
    db.execute(text("DELETE FROM inference_images WHERE job_id=:job_id"), {"job_id": job_id})

    # crop 생성 + DB 저장
    for r in det_rows:
        di = int(r["det_index"])
        x1, y1, x2, y2 = float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])
        clamped = _clamp_xyxy(x1, y1, x2, y2, w, h)
        if not clamped:
            continue
        x1i, y1i, x2i, y2i = clamped

        crop_img = img.crop((x1i, y1i, x2i, y2i))
        tmp = f"/tmp/crop_{job_id}_{di}_{uuid.uuid4().hex}.jpg"
        crop_img.save(tmp, "JPEG", quality=90)

        object_key = f"{job_id}/{di}.jpg"
        try:
            minio.fput_object(crop_bucket, object_key, tmp, content_type="image/jpeg")
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

        db.execute(text("""
            INSERT INTO crops(
                job_id, photo_id, det_index,
                class_id, class_name, score,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                bucket, object_key
            )
            VALUES(
                :job_id, :photo_id, :det_index,
                :class_id, :class_name, :score,
                :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                :bucket, :object_key
            )
        """), {
            "job_id": job_id,
            "photo_id": photo_id,
            "det_index": di,
            "class_id": int(r["class_id"]),
            "class_name": r["class_name"] or str(r["class_id"]),
            "score": float(r["score"]),
            "bbox_x1": float(x1i),
            "bbox_y1": float(y1i),
            "bbox_x2": float(x2i),
            "bbox_y2": float(y2i),
            "bucket": crop_bucket,
            "object_key": object_key,
        })

    # pred 이미지 생성 (탐지 결과 존재할 때만)
    dets = []
    for r in det_rows:
        di = int(r["det_index"])
        dets.append(Det(
            det_index=di,
            class_id=int(r["class_id"]),
            class_name=(r["class_name"] or str(r["class_id"])),
            score=float(r["score"]),
            bbox_xyxy=(float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])),
            geometry=seg_map.get(di),
        ))

    pred_img = draw_yolo_style(img, dets)
    pred_tmp = f"/tmp/pred_{job_id}_{uuid.uuid4().hex}.jpg"
    pred_img.save(pred_tmp, "JPEG", quality=90)

    pred_key = f"{job_id}/pred.jpg"
    try:
        minio.fput_object(pred_bucket, pred_key, pred_tmp, content_type="image/jpeg")
    finally:
        try:
            os.remove(pred_tmp)
        except Exception:
            pass

    db.execute(text("""
        INSERT INTO inference_images(job_id, photo_id, bucket, object_key)
        VALUES(:job_id, :photo_id, :bucket, :object_key)
    """), {
        "job_id": job_id,
        "photo_id": photo_id,
        "bucket": pred_bucket,
        "object_key": pred_key,
    })

    return len(det_rows)

