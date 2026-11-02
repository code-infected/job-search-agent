FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .

RUN useradd --create-home --uid 10001 agent && mkdir -p /app/data && chown agent /app/data
USER agent

EXPOSE 8000
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
