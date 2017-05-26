import sys, os  

sys.path.append(os.getcwd())
from config import *

sys.path.append(PROJ_ROOT + "src/")

import sframe as sf
import csv

def modify_csv_rows(input_url=PROJ_ROOT+'data/data.csv', output_url=PROJ_ROOT+'data/modified_data.csv'):
  print("Transforming initial data csv")
  with open(output_url, 'w') as new:
    newWriter = csv.writer(new, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    with open(input_url) as csvfile:
      newWriter.writerow(['Source Port', 'Destination Port', 'Score', 'Source', 'Destination', 'Protocol', 'IP_Flags', 'Length', 'Protocols in frame', 'Time', 'tcp_Flags', 'TCP Segment Len', 'udp_Length'])
      first = True
      for row in csv.reader(csvfile, delimiter=',', quotechar='"'):
        if first == True:
          first = False
          continue
        if row[2]:
          row.pop(0)
          row.pop(0)
        else:
          row.pop(2)
          row.pop(2)
        other = row[2]
        if (other[0] == other [1]) and (other[1] == other[3]) and (other[3] == other[4]) and (other[4] == other[6]) and (other[6] == other[7]):
          row[2] = 1
        else:
          row[2] = 0
        newWriter.writerow(row) 
      print("Csv row modification complete\n##############################")

def flow_id(x):
    if x['Source']>x['Destination']:
        return x['Source']+'-'+x['Destination']+'-'+str(x['Source Port'])+'-'+str(x['Destination Port'])+'-'+str(x['Protocol'])
    else:
        return x['Destination']+'-'+x['Source']+'-'+str(x['Destination Port'])+'-'+str(x['Source Port'])+'-'+str(x['Protocol'])

def flow_separator(input_url=PROJ_ROOT+'data/modified_data.csv', output_url=PROJ_ROOT+"models/sorted_flow.csv"):

  print("Initiating flow separation")
  sorted_flow = sf.SFrame.read_csv(input_url,verbose=False)
    
  print("Preprocessing file")
  # Preprocess file
  sorted_flow = sorted_flow[(sorted_flow['Source Port']!='')&(sorted_flow['Destination Port']!='')]
  sorted_flow['Forward'] = sorted_flow.apply(lambda x: 1 if x['Source']>x['Destination'] else 0 )
  sorted_flow['tcp_Flags'] = sorted_flow['tcp_Flags'].apply(lambda x:int(x,16) if x!='' else 0)
  sorted_flow['UFid'] = sorted_flow.apply(lambda x:flow_id(x))
  sorted_flow = sorted_flow.sort(['UFid','Time'])

  # Master flow list
  Flow = [] 

  # Incremental vars
  current_flow_id = 0 # incrementing id for flow
  prev_flow_id = None
  startTime = None   ##Start Time of each flow to implement timeout

  for row in sorted_flow:
    
    # Means prev is set to none so no previous flow to continue
    if prev_flow_id is None:
      if startTime is None:
        startTime = row['Time']
      # Add this new flow to the current_flow
      Flow.append(current_flow_id)
      prev_flow_id = row['UFid']
      
    elif (row['UFid'] == prev_flow_id):
      # TCP termination
      if row['tcp_Flags']&1:
        Flow.append(current_flow_id)
        prev_flow_id = None
        startTime = None
        current_flow_id += 1

      # Timeout termination and restart
      elif row['Time']-startTime>=3600:
        current_flow_id = current_flow_id + 1
        Flow.append(current_flow_id)
        prev_flow_id = None
        startTime = row['Time']

      # New time
      else:
        Flow.append(current_flow_id)
        prev_flow_id = row['UFid']

    else:
      # Previous Flow tuple didnt receive any more packets, start a new flow
      current_flow_id = current_flow_id + 1
      Flow.append(current_flow_id)
      prev_flow_id = row['UFid']
      startTime = row['Time']

  print("Flow sorting complete")
  print(len(sf.SArray(Flow).unique()))
  sorted_flow['Flow'] = sf.SArray(Flow)
  temp = sorted_flow.groupby('Flow',{
        'Count':sf.aggregate.COUNT()
      })
  sorted_flow['FlowNo.'] = sf.SArray(Flow)
  sorted_flow.save(output_url)
  print("Flow saved\n##############################")

modify_csv_rows()
flow_separator()
print("Raw flows are ready\n##############################")
