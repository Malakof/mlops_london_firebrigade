FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils python3-pip python3.8-venv
RUN ls -l
WORKDIR /app
RUN ls -l
COPY . .
#COPY ./src/api/api_gateway/requirements.txt .
#COPY ./src/api/api_gateway/main.py .
#COPY ./src/api/api_gateway/utils/__init__.py .
#COPY ./src/api/api_gateway/utils/config.py .
RUN python3 -m venv ./venv
RUN . venv/bin/activate
RUN pip install --index-url https://test.pypi.org/simple/ --no-cache-dir -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]