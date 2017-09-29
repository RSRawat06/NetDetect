# Flow featurization

This software is for generating datasets that you can download using
`python3 -m botnet_attention.iscx.download` and 
`python3 -m botnet_attention.isot.download`.

Unless you intend on adding a new dataset or modifying the preprocessing procedures for the default ISCX and ISOT datasets, you should not need to use this module.


Primary: 
java -Xmx3g -XX:-UseGCOverheadLimit -Djava.library.path=/service/CICFlowMeter/jnetpcap -jar /service/CICFlowMeter/CICFlowMeter.jar /service/CICFlowMeter/input/ /service/CICFlowMeter/output/
bash /service/utils/split_pcap.sh /service/CICFlowMeter/input/ISCX_Botnet-Training.pcap /service/CICFlowMeter/input/output.pcap
