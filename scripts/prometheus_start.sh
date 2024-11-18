docker run -d -p 9091:9091 prom/pushgateway
prometheus --config.file=./prometheus.yml --storage.tsdb.path=../data/prometheus
                