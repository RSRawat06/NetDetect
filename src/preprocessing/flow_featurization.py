import sys, os  

sys.path.append(os.getcwd())
from config import *

sys.path.append(PROJ_ROOT + "src/")

import sframe as sf

def flow_featurization(input_url=PROJ_ROOT+"models/sorted_flow.csv", output_url=PROJ_ROOT+"models/all_features.csv"):
  print("Intializing flow featurization")

  flow_list = sf.SFrame.read_csv(input_url,verbose=False)
  
  ## Ratio of incoming to outgoing packets
  temp = flow_list.groupby('FlowNo.',{
      'NumForward' : sf.aggregate.SUM('Forward'),
      'Total' : sf.aggregate.COUNT()
    })
  temp['IOPR']= temp.apply(lambda x: ((x['Total']-x['NumForward'])*1.0)/x['NumForward'] if x['NumForward'] !=0 else (-1) )
  temp = temp['FlowNo.','IOPR']

  flow_list = flow_list.join(temp,on='FlowNo.')
  del(temp)
  print(" Ratio measuring complete")

  ## First Packet Length
  FlowFeatures = ['Source','Destination','Source Port','Destination Port','Protocol']
  FPL = flow_list.groupby(['FlowNo.'],{
      'Time':sf.aggregate.MIN('Time')
    })
  FPL = FPL.join(flow_list,on =['FlowNo.','Time'])[['FlowNo.','Length']].unique()
  FPL = FPL.groupby(['FlowNo.'],{
      'FPL':sf.aggregate.AVG('Length')
    })
  flow_list = flow_list.join(FPL, on ='FlowNo.')
  del(FPL)
  print(" Packet length measured")

  ## Number of packets per flow
  temp = flow_list.groupby(['FlowNo.'],{
      'NumPackets':sf.aggregate.COUNT()
    })
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Packet sum measured")

  ## Number of bytes exchanged
  temp = flow_list.groupby(['FlowNo.'],{
      'BytesEx':sf.aggregate.SUM('Length')
    })
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Byte exchange measured")

  ## Standard deviation of packet length
  temp = flow_list.groupby(['FlowNo.'],{
      'StdDevLen':sf.aggregate.STDV('Length')
    })
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Standard deviation of packet length measured")

  ## Same length packet ratio
  temp2 = flow_list.groupby(['FlowNo.'],{
      'SameLenPktRatio':sf.aggregate.COUNT_DISTINCT('Length')
    })
  temp = flow_list.groupby(['FlowNo.'],{
      'NumPackets':sf.aggregate.COUNT()
    })
  temp = temp.join(temp2,on='FlowNo.')
  temp['SameLenPktRatio'] = temp['SameLenPktRatio']*1.0/temp['NumPackets']
  temp2 = None
  temp = temp[['FlowNo.','SameLenPktRatio']]
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Same length packet ratio measured")

  ## Duration of flow
  timeF = flow_list.groupby(['FlowNo.'],{
      'startTime':sf.aggregate.MIN('Time'),
      'endTime':sf.aggregate.MAX('Time')
    })
  timeF['Duration'] = timeF['endTime'] - timeF['startTime']
  timeF = timeF[['FlowNo.','Duration']]
  flow_list = flow_list.join(timeF, on ='FlowNo.')
  print("  Duration of flow measured")

  # Relevant Features extracted till now
  features = ['BytesEx',
   'Destination',
   'Destination Port',
   'Duration',
   'FPL',
   'IP_Flags',
   'Length',
   'NumPackets',
   'Protocol',
   'Protocols in frame',
   'SameLenPktRatio',
   'Score',
   'Source',
   'Source Port',
   'StdDevLen',
   'TCP Segment Len',
   'Time',
   'tcp_Flags',
   'FlowNo.',
   'udp_Length',
   'IOPR']
  flow_list = flow_list[features]

  ## Average packets per second
  temp =  flow_list.groupby(['FlowNo.'],{
      'NumPackets':sf.aggregate.COUNT()
    })
  temp = temp.join(timeF,on=['FlowNo.'])
  temp['AvgPktPerSec'] = temp.apply(lambda x:0.0 if x['Duration'] == 0.0 else x['NumPackets']*1.0/x['Duration'])
  temp = temp[['FlowNo.','AvgPktPerSec']]
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Average packets calculated")

  ##Average Bits Per Second
  temp = flow_list.groupby(['FlowNo.'],{
      'BytesEx':sf.aggregate.SUM('Length')
    })
  temp = temp.join(timeF,on=['FlowNo.'])
  temp['BitsPerSec'] = temp.apply(lambda x:0.0 if x['Duration'] == 0.0 else x['BytesEx']*8.0/x['Duration'])
  temp = temp[['FlowNo.','BitsPerSec']]
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Average bits calculated")

  ## Average Packet Lentgth
  temp = flow_list.groupby(['FlowNo.'],{
      'APL':sf.aggregate.AVG('Length')
    })
  flow_list = flow_list.join(temp, on ='FlowNo.')
  del(temp)
  print(" Average package length calculated")

  flow_list['IAT'] = 0
  flow_list = flow_list.sort(['FlowNo.','Time'])
  prev = None
  prevT = None
  li = []
  for x in flow_list:
    if prev is None or x['FlowNo.']!= prev:
      li.append(0)
    else:
      li.append(x['Time']-prevT)    
    prev = x['FlowNo.']
    prevT = x['Time']
  flow_list['IAT'] = sf.SArray(li)

  ## Null Packets handling
  def checkNull(x):
    if(x['TCP Segment Len']=='0' or x['udp_Length']==8 ):
      return 1
    elif('ipx' in x['Protocols in frame'].split(':')):
      l = x['Length'] - 30
      if('eth' in x['Protocols in frame'].split(':')):
        l = l - 14
      if('ethtype' in x['Protocols in frame'].split(':')):
        l = l - 2
      if('llc' in x['Protocols in frame'].split(':')):
        l = l - 8
      if(l==0 or l==-1):
        return 1
    return 0
  flow_list['isNull'] = flow_list.apply(lambda x:checkNull(x))
  NPEx = flow_list.groupby(['FlowNo.'],{
      'NPEx':sf.aggregate.SUM('isNull')
    })
  flow_list = flow_list.join(NPEx, on ='FlowNo.')
  del(NPEx)
  print("  Null packets handled")
  
  flow_list['Forward'] = flow_list.apply(lambda x: 1 if x['Source']>x['Destination'] else 0 )
  temp = flow_list.groupby('FlowNo.',{
      'NumForward' : sf.aggregate.SUM('Forward'),
    })

  flow_list= flow_list.join(temp,on='FlowNo.')
  del(temp)
  
  flow_list = flow_list.groupby('FlowNo.',{
      'BytesEx' : sf.aggregate.SELECT_ONE('BytesEx'),
      'Destination' : sf.aggregate.SELECT_ONE('Destination'),
      'Destination Port' : sf.aggregate.SELECT_ONE('Destination Port'),
      'Duration' : sf.aggregate.SELECT_ONE('Duration'),
      'FPL' : sf.aggregate.SELECT_ONE('FPL'),
      'IP_Flags' : sf.aggregate.SELECT_ONE('IP_Flags'),
      'Length' : sf.aggregate.SELECT_ONE('Length'),
      'NumPackets' : sf.aggregate.SELECT_ONE('NumPackets'),
      'Protocol' : sf.aggregate.SELECT_ONE('Protocol'),
      'Protocols in frame' : sf.aggregate.SELECT_ONE('Protocols in frame'),
      'SameLenPktRatio' : sf.aggregate.SELECT_ONE('SameLenPktRatio'),
      'Score' : sf.aggregate.SELECT_ONE('Score'),
      'Source' : sf.aggregate.SELECT_ONE('Source'),
      'Source Port' : sf.aggregate.SELECT_ONE('Source Port'),
      'StdDevLen' : sf.aggregate.SELECT_ONE('StdDevLen'),
      'IAT' : sf.aggregate.SELECT_ONE('IAT'),
      'isNull' : sf.aggregate.SELECT_ONE('isNull'),
      'NPEx' : sf.aggregate.SELECT_ONE('NPEx'),
      'APL' : sf.aggregate.SELECT_ONE('APL'),
      'BitsPerSec' : sf.aggregate.SELECT_ONE('BitsPerSec'),
      'AvgPktPerSec' : sf.aggregate.SELECT_ONE('AvgPktPerSec'),
      'udp_Length' : sf.aggregate.SELECT_ONE('udp_Length'),
      'tcp_Flags' : sf.aggregate.SELECT_ONE('tcp_Flags'),
      'Time' : sf.aggregate.SELECT_ONE('Time'),
      'TCP Segment Len' : sf.aggregate.SELECT_ONE('TCP Segment Len'),
      'IOPR' : sf.aggregate.SELECT_ONE('IOPR'),
      'NumForward' : sf.aggregate.SELECT_ONE('NumForward')
    })
  flow_list.save(output_url)
  print("Flow feature generation complete")
  print("Updated flow saved")

flow_featurization()
