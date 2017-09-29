# Flow featurization

This container is for generating datasets that you can download using
`python3 -m NetDetect.datasets.iscx.download` in the other container.

Unless you intend on adding a new dataset or modifying the preprocessing procedures for the default ISCX and ISOT datasets, you should not need to use this module.

Use CIC Flowmeter:
```
bash /featurizer/utils/split_pcap.sh /featurizer/CICFlowMeter/input/ISCX_Botnet-Training.pcap /featurizer/CICFlowMeter/input/output.pcap
java -Xmx3g -XX:-UseGCOverheadLimit -Djava.library.path=/featurizer/CICFlowMeter/jnetpcap -jar /featurizer/CICFlowMeter/CICFlowMeter.jar /featurizer/CICFlowMeter/input/ /featurizer/CICFlowMeter/output/
```

