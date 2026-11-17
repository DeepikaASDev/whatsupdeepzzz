FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deepzzzz ./deepzzzz

RUN useradd --create-home appuser && mkdir -p /app/data && chown appuser /app/data
USER appuser

VOLUME /app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "deepzzzz.api:app", "--host", "0.0.0.0", "--port", "8000"]
