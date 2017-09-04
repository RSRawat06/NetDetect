# Botnet Attention
Botnet detection in enterprise networks using recurrent neural networks with attention mechanisms (prototype: 9/1/2017)

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

### What's special
* Applying recurrent networks to not only sequences of packets, but higher level sequences of flows
* Using attention mechanisms to better identify patterns hiding in the relationships between packets and flows
* Leverage end-to-end differentiability to learn minimally supervised packet-level encoding

### Installation
To get Docker started:

`docker-compose up --build`

`bash access_cluster.sh`

To download models:

`python3 -m botnet_attention.isot.download`

`python3 -m botnet_attention.iscx.download`

To run training:

`python3 -m botnet_attention.isot.train`

`python3 -m botnet_attention.iscx.train`

### Overview
This codebase offers:
* Download and loading scripts for ISOT and ISCX datasets
* Automated preprocessing, including pcap -> csv conversions
* Feature extraction tools for packets
* Network flow segmentation and feature extraction for flows
* Three different recurrent neural network architectures implemented in Tensorflow
* Vanilla RNN model to predict infection status given a sequence of packets in a flow
* Deep RNN model to predict infection status given a sequence of network flows which each
compose a sequence of packets
* Self-attention RNN to predict infection status given a sequence of network flows which each
compose a sequence of packets


### Contributions
Feel free to make a pull request if you have any suggestions.
Contributors are listed under `contributors.txt`.

# License
MIT License

Copyright (c) 2017 Eric Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

