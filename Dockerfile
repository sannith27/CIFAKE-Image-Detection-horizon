FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    WEB_CONCURRENCY=1 \
    CIFAKE_ENABLE_GRADCAM=0 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-web.txt ./requirements-web.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-web.txt

COPY . .

RUN mkdir -p database static/uploads static/gradcam

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 1 --worker-class sync --timeout 180 --graceful-timeout 30 --max-requests 50 --max-requests-jitter 10 app:app"]
