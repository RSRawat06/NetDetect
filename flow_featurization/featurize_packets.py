"""
This script offers functions to featurize flows.
"""

import csv

headers = ['Source Port', 'Destination Port', 'Score', 'Source', 'Destination', 'Protocol', 'IP_Flags', 'Length', 'Protocols in frame', 'Time', 'tcp_Flags', 'TCP Segment Len', 'udp_Length']


def score_packets(input_url='data/raw_packets.csv', output_url='data/scored_packets.csv'):
  '''
  Adds score indicators to botnets
  '''
  print("Transforming initial data csv")
  with open(output_url, 'w') as raw_flows:
    writer = csv.writer(raw_flows, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    with open(input_url) as csvfile:
      writer.writerow(headers + "Score")
      first = True
      for row in csv.reader(csvfile, delimiter=',', quotechar='"'):
        if first is True:
          first = False
          continue
        if row[headers.index('Label')] == "BENIGN":
          row.append(0)
        else:
          row.append(1)
        writer.writerow(row)


def featurize_packets(input_url='data/scored_packets.csv', output_url='data/featurized_packets.csv'):
  '''
  Featurizes packets
  '''
  pass
