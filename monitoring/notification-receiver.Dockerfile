FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY notification_receiver.py .
USER 65532:65532
EXPOSE 9001
CMD ["python", "notification_receiver.py"]
