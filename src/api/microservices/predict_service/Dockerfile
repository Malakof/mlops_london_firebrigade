FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils python3-pip
WORKDIR /app
COPY ./src/api/microservices/predict_service/requirements.txt .
COPY ./src/api/microservices/predict_service/main.py .

COPY ./src/utils ./src/utils
COPY ./src/model ./src/model

RUN pip3 install --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]