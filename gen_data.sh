source ~/toolkit/2python/venv/bin/activate
bash clean_training_data.sh
bash src/preprocessing/csvify.sh data/$1 data/data.csv
python src/preprocessing/flow_generator.py
python src/preprocessing/flow_featurization.py
tail -n +2 models/all_features.csv >> models/new_complete.csv

