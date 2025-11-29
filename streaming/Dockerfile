FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    requests==2.31.0

COPY stream_simulator.py .

CMD ["python", "stream_simulator.py"]