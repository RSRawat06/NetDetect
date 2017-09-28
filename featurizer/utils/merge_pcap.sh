# Merge pcap files
# Usage: bash merge_pcap.sh merged.pcap

echo "Merging PCAP"
mergecap *.pcap -w $1
echo "Merge complete"
