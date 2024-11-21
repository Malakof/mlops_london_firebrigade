FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils python3-pip
WORKDIR /app
COPY ./src/api/api_gateway/requirements.txt .
COPY ./src/api/api_gateway/main.py .
RUN mkdir -p src
RUN mkdir -p src/utils
COPY ./src/utils/ /src/utils/

RUN pip3 install --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]