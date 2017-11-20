'''
Standard configuration for iscx dataset.
'''

import os


DUMPS_DIR = os.path.dirname(os.path.realpath(__file__)) + "/dumps/"

RAW_TRAIN_NAME = "ISCX_Botnet-Training.pcap_ISCX.csv"
RAW_TEST_NAME = "ISCX_Botnet-Testing.pcap_ISCX.csv"

PROCESSED_TRAIN_NAME = "train_dataset.np"
PROCESSED_TEST_NAME = "test_dataset.np"

TRAIN_FEATURES_URL = ""
TRAIN_LABELS_URL = ""
TEST_FEATURES_URL = ""
TEST_LABELS_URL = ""

participant_fields = ["Source_IP", "Destination_IP"]
numerical_fields = [
    "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets", "Total_Length_of_Bwd_Packets",
    "Fwd_Packet_Length_Max", "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Mean",
    "Fwd_Packet_Length_Std", "Bwd_Packet_Length_Max", "Bwd_Packet_Length_Min",
    "Bwd_Packet_Length_Mean", "Bwd_Packet_Length_Std", "Flow_Bytes/s",
    "Flow_Packets/s", "Flow_IAT_Mean", "Flow_IAT_Std",
    "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Total", "Fwd_IAT_Mean",
    "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Total",
    "Bwd_IAT_Mean", "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min",
    "Fwd_PSH_Flags", "Bwd_PSH_Flags", "Fwd_URG_Flags", "Bwd_URG_Flags",
    "Fwd_Header_Length", "Bwd_Header_Length", "Fwd_Packets/s", "Bwd_Packets/s",
    "Min_Packet_Length", "Max_Packet_Length", "Packet_Length_Mean",
    "Packet_Length_Std", "Packet_Length_Variance", "FIN_Flag_Count",
    "SYN_Flag_Count", "RST_Flag_Count", "PSH_Flag_Count", "ACK_Flag_Count",
    "URG_Flag_Count", "CWE_Flag_Count", "ECE_Flag_Count", "Down/Up_Ratio",
    "Average_Packet_Size", "Avg_Fwd_Segment_Size", "Avg_Bwd_Segment_Size",
    "Fwd_Header_Length", "Fwd_Avg_Bytes/Bulk", "Fwd_Avg_Packets/Bulk",
    "Fwd_Avg_Bulk_Rate", "Bwd_Avg_Bytes/Bulk", "Bwd_Avg_Packets/Bulk",
    "Bwd_Avg_Bulk_Rate", "Subflow_Fwd_Packets", "Subflow_Fwd_Bytes",
    "Subflow_Bwd_Packets", "Subflow_Bwd_Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active_Mean", "Active_Std", "Active_Max", "Active_Min", "Idle_Mean",
    "Idle_Std", "Idle_Max", "Idle_Min"
]
malicious_ips = [
    "192.168.2.112", "131.202.243.84", "192.168.5.122", "198.164.30.2",
    "192.168.2.110", "192.168.5.122", "192.168.4.118", "192.168.5.122",
    "192.168.2.113", "192.168.5.122", "192.168.1.103", "192.168.5.122",
    "192.168.4.120", "192.168.5.122", "192.168.2.112", "192.168.2.110",
    "192.168.2.112", "192.168.4.120", "192.168.2.112", "192.168.1.103",
    "192.168.2.112", "192.168.2.113", "192.168.2.112", "192.168.4.118",
    "192.168.2.112", "192.168.2.109", "192.168.2.112", "192.168.2.105",
    "192.168.1.105", "192.168.5.122", "147.32.84.180", "147.32.84.170",
    "147.32.84.150", "147.32.84.140", "147.32.84.130", "147.32.84.160",
    "10.0.2.15", "192.168.106.141", "192.168.106.131", "172.16.253.130",
    "172.16.253.131", "172.16.253.129", "172.16.253.240", "74.78.117.238",
    "158.65.110.24", "192.168.3.35", "192.168.3.25", "192.168.3.65",
    "172.29.0.116", "172.29.0.109", "172.16.253.132", "192.168.248.165",
    "10.37.130.4"
]

