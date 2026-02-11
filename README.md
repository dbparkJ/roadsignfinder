# roadsignfinder

main start code : `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
inference start code : `celery -A inference_worker.celery_app worker -Q inference --loglevel=info`
sam3 start code : `celery -A sam3_worker.celery_app worker -Q sam3 --pool=threads --concurrency=1 --loglevel=info`