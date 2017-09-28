# Split pcap files
# Usage: bash split_pcap.sh big.pcap child_base_name.pcap

echo "Splitting PCAP"
editcap -c 3500000 $1 $2
echo "Split complete"
