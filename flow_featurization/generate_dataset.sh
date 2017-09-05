# Generate a fully featurized dataset
# Usage: bash ./generate_dataset.sh RAW_DATASET_NAME
# ^ Assuming that RAW_DATASET_NAME is under the 
# ./data/ directory. 

echo "Beginning dataset generation"
bash utils/pcap_to_csv.sh data/$1 data/raw_packets.csv
echo "Success: pcap data file translated to csv."

echo "Preprocessing dataset"
python featurize_packets.py
python featurize_flows.py
echo "Preprocessing complete"

