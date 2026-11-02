FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py config.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY ["Data/processed Data/", "./Data/processed Data/"]
COPY Output/ ./Output/

RUN mkdir -p /app/uploads /app/Output/models \
    && adduser --disabled-password --gecos "" --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--access-logfile", "-", "app:app"]
