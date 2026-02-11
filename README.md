# RoadSign Finder

RoadSign Finder는 도로 표지판 이미지를 업로드하면 YOLO 기반 객체 검출, OCR, SAM3 후처리까지 비동기 워커로 연결해 결과를 반환하는 추론 파이프라인 프로젝트입니다.

## What It Does

- FastAPI 서버에서 이미지 업로드/조회 API 제공
- MinIO를 원본 및 추론 결과 스토리지로 사용
- `inference_worker`에서 YOLO 추론 수행
- YOLO 박스 기준 crop 이미지 생성 후 `crop` 버킷 업로드
- crop 이미지에 대해 PaddleOCR 실행 후 JSON 저장/반환
- `sam3_worker`에서 추가 분류/후처리 수행

## Project Layout

- `fastAPI_Server`: 메인 API 서버
- `inference_worker`: YOLO + crop + OCR 워커
- `sam3_worker`: SAM3 후처리 워커
- `paddleocr`: OCR 유틸/워커 코드
- `requirements.txt`: 전체 의존성
- `requirements.post.txt`: 설치 후 `numpy==2.3.1` 재고정 용도

## Installation

Python 가상환경을 활성화한 뒤 아래 순서로 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --force-reinstall --no-deps -r requirements.post.txt
```

현재 `requirements.txt`에는 아래 버전이 명시되어 있습니다.

- `torch==2.7.1+cu126`
- `paddlepaddle-gpu==3.2.1`
- `paddleocr[doc-parser]`

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

## Inference Flow

1. 사용자가 이미지를 업로드하면 MinIO에 저장됩니다.
2. API가 `inference_worker` 작업을 큐에 등록합니다.
3. 워커가 MinIO에서 원본을 내려받아 YOLO 추론을 수행합니다.
4. 검출 박스를 crop 이미지로 저장하고 MinIO `crop` 버킷에 업로드합니다.
5. crop 이미지로 OCR을 수행해 JSON을 저장하고 결과 payload에 포함합니다.
6. 필요 시 `sam3_worker`가 이어서 후처리를 수행합니다.
