bash /CICFlowMeter/utils/split_pcap.sh /CICFlowMeter/input/iscx.pcap /CICFlowMeter/input/output.pcap
java -Xmx3g -XX:-UseGCOverheadLimit -Djava.library.path=/CICFlowMeter/jnetpcap -jar /CICFlowMeter/CICFlowMeter/CICFlowMeter.jar /CICFlowMeter/input/ /CICFlowMeter/output/
