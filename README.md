# RoadSign Finder

RoadSign Finder는 도로 표지판 이미지를 업로드하면 YOLO 기반 객체 검출, OCR, SAM3 후처리까지 비동기 워커로 연결해 결과를 반환하는 추론 파이프라인 프로젝트입니다.

## What It Does

- FastAPI 서버에서 이미지 업로드/조회 API 제공
- MinIO를 원본 및 추론 결과 스토리지로 사용
- `inference_worker`에서 YOLO 추론 + crop 생성/업로드 수행
- YOLO 박스 기준 crop 이미지 생성 후 `crop` 버킷 업로드
- `ocr_worker`가 crop 이미지를 별도 환경에서 OCR 처리
- `sam3_worker`에서 추가 분류/후처리 수행

## Project Layout

- `fastAPI_Server`: 메인 API 서버
- `inference_worker`: YOLO + crop 업로드 워커 (Torch 환경)
- `sam3_worker`: SAM3 후처리 워커
- `ocr_worker`: OCR 워커 (Paddle 환경)
- `requirements.server.txt`: API 서버용 의존성
- `requirements.torch-workers.txt`: inference/sam3 워커용 의존성
- `requirements.ocr-worker.txt`: OCR 워커용 의존성

## Installation

권장 구성은 환경 분리입니다.

1. `server` 환경: API 서버
2. `torch-workers` 환경: `inference_worker`, `sam3_worker`
3. `ocr-worker` 환경: `ocr_worker` (Paddle 전용)

### 1) API Server Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.server.txt
```

### 2) Torch Workers Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.torch-workers.txt
```

### 3) OCR Worker Environment (Paddle)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.ocr-worker.txt
```

필요 시 NumPy를 고정하려면:

```bash
python -m pip install --force-reinstall --no-deps numpy==2.3.1
```

참고 버전:

- Torch: `torch==2.7.1+cu126`
- Paddle: `paddlepaddle-gpu==3.2.1`
- OCR: `paddleocr[doc-parser]`

## Run

각 프로세스를 별도 터미널에서 실행합니다.

```bash
# API
cd fastAPI_Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Inference worker
celery -A inference_worker.celery_app worker -Q inference --loglevel=info
```

```bash
# SAM3 worker
celery -A sam3_worker.celery_app worker -Q sam3 --pool=threads --concurrency=1 --loglevel=info
```

```bash
# OCR worker
celery -A ocr_worker.celery_app worker -Q ocr --loglevel=info
```

## Inference Flow

1. 사용자가 이미지를 업로드하면 MinIO에 저장됩니다.
2. API가 `inference_worker` 작업을 큐에 등록합니다.
3. 워커가 MinIO에서 원본을 내려받아 YOLO 추론을 수행합니다.
4. 검출 박스를 crop 이미지로 저장하고 MinIO `crop` 버킷에 업로드합니다.
5. API가 후속 작업으로 `sam3_worker`와 `ocr_worker`를 각각 큐에 등록합니다.
6. `ocr_worker`는 MinIO crop 객체를 내려받아 OCR JSON을 생성/저장하고 callback으로 결과를 반영합니다.
7. `sam3_worker`는 후처리 결과를 callback으로 반영합니다.
