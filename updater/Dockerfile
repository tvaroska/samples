FROM python:3.11-slim

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV APP_DIR=/app
WORKDIR $APP_DIR
COPY * /$APP_DIR/

ENTRYPOINT ["python", "process.py"]
