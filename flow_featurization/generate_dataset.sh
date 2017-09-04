echo "You must be running this file from within the file directory"
echo "assert that ./ == flow_featurization"
sleep 5

echo "Beginning dataset generation"
bash utils/pcap_to_csv.sh data/$1 data/raw_packets.csv
echo "Success: pcap data file translated to csv."

echo "Preprocessing dataset"
python3 preprocess_dataset.py
echo "Preprocessing complete"

