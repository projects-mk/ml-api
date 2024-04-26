FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install -r requirements.txt
ENV GIT_PYTHON_REFRESH=quiet

EXPOSE 8000
COPY . .

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]