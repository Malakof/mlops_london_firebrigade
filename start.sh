#!/usr/bin/env bash

## RUN prom/pushgateway
docker rm prom/pushgateway  
docker pull prom/pushgateway  
# check ports
docker run -d -p 9091:9091 prom/pushgateway
#To check metrics list http://localhost:9091/#metrics
## RUN grafana
docker rm grafana/grafana-enterprise
docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise
#Once Grafana is up the you can navigate to: http://localhost:3000

## RUN `build_features_service` latest
docker pull orvm/mlops_firebrigade:build_features_service-latest
docker run orvm/mlops_firebrigade:build_features_service-latest
#Once build_features is UP then you can navigate to http://localhost:8002


## RUN `process_data_service` latest
docker pull orvm/mlops_firebrigade:process_data_service-latest
docker run orvm/mlops_firebrigade:process_data_service-latest
#Once process_data_service is UP then you can navigate to http://localhost:8001


## RUN `train_model_service` latest
docker pull orvm/mlops_firebrigade:train_model_service-latest
docker run orvm/mlops_firebrigade:train_model_service-latest
#Once train_model is UP then you can navigate to http://localhost:8003


## RUN `predict_service` latest
docker pull orvm/mlops_firebrigade:predict_service-latest
docker run orvm/mlops_firebrigade:predict_service-latest
#Once process_data_service is UP then you can navigate to http://localhost:8004



## AT the end RUN `api_gateway` latest
docker pull orvm/mlops_firebrigade:api_gateway-latest
docker run orvm/mlops_firebrigade:api_gateway-latest
#Once api_gateway is UP then you can navigate to http://localhost:8000
