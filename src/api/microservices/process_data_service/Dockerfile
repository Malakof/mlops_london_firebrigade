# Use the custom base image
FROM base:latest
# RUN apt-get update && apt-get install -y python3.12 python3.12-distutils python3-pip
WORKDIR /app
COPY ./src/api/microservices/process_data_service/requirements.txt .
COPY ./src/api/microservices/process_data_service/main.py .

COPY ./src/utils ./src/utils
COPY ./src/data ./src/data

RUN pip3 install --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]