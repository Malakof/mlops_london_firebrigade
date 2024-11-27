# Use the custom base image
FROM myapp/base:latest
# RUN apt-get update && apt-get install -y python3.12 python3.12-distutils python3-pip
WORKDIR /app
COPY ./src/api/api_gateway/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "0.0.0.0", "--port", "9092"]