git clone https://github.com/Malakof/mlops_london_firebrigade/
cd mlops_london_firebrigade
docker-compose up --build -d
sleep 5
./scripts/train_model.sh
