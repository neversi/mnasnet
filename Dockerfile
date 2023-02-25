FROM zironycho/pytorch:1.6.0-slim-py3.7-v1
WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "80"]