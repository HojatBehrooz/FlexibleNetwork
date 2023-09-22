# FlexibleNetwork
**About**

In this repository, we present a novel method for evacuation route planning based on our research study. The study introduces an innovative approach for emergency evacuation routing that adapts dynamically to unforeseen disruptions in road networks. It utilizes a dynamic graph-based algorithm, employing Depth First Search (DFS), to distribute traffic flow across the network. Notably, the algorithm determines road segment directions to enhance evacuation performance. The proposed method was implemented in an illustrative network example, and its performance was empirically compared with Dinic's algorithm. The results indicated that the proposed method in this study outperforms Dinic's algorithm regarding the total number of evacuees.

**Getting Started**

Our study outlines a systematic approach for adaptive emergency evacuation routing, framing it as a graph-based maximum flow problem. The source nodes represent dangerous areas, and the sink nodes are safe zones, aiming to dynamically allocate traffic flow and link directions to maximize network throughput from sources to sinks. The method consists of four key steps: graph initialization, a static version of the Quanta Adaptive Routing (QAR) algorithm, extension to dynamic scenarios, and empirical evaluation against Dinic's algorithm.

The process for adaptively determining maximum flow in emergency evacuation begins by initializing a road network as a graph (G(V, E)), where nodes (V) and links (E) have attributes of maximum capacity and travel time. The algorithm dynamically assigns directions to links to improve total flow, effectively creating reverse links with identical attributes for each existing link. The Quanta Adaptive Routing (QAR) algorithm, originally designed for graphs with one source and one sink node, is extended for cases with multiple source and sink nodes. This extension introduces dummy super source and sink nodes to consolidate the network into a single source-sink structure, allowing the application of the QAR algorithm.

The Quanta Adaptive Routing (QAR) algorithm is an iterative and recursive process used to direct traffic flow within a graph. Each iteration aims to move a specified amount of flow from the source node to the sink node, with the algorithm terminating when the actual flow reaching the sink node falls below a minimum threshold (epsilon). The flow distribution is achieved through a procedure known as the Push function, which takes inputs including the origin node (holding the flow), the destination node (always the sink node), the target flow (current flow at the origin node), and a list of visited nodes for the current iteration.

The Push function operates recursively, starting with the source node as the initial origin and the source node's flow as the initial target flow. It progressively directs the flow through the graph, updating the origin node to the successor node that receives the largest portion of the flow, adjusting the target flow accordingly, and updating the list of visited nodes. This recursive process continues until the origin node reaches the sink node, forming a path. The Push function utilizes a Depth First Search (DFS) strategy to explore the graph, moving as deeply as possible along each branch before backtracking.

The actual flow assigned to each link in the path is determined based on the flow reaching the sink node in that iteration. Any remaining flow is distributed among the successor nodes of the source node in the next iteration. This process continues until the termination condition is met. Additionally, when the flow reaches a node, it is distributed among the outgoing links based on the flow history of those links in previous rounds.

**Illustrative Example and Validation**

An illustrative example was conducted to validate the proposed Quanta Adaptive Routing (QAR) algorithm's performance. The study implemented the algorithm on a grid road network comprising 25 nodes and 40 links. In this example, nodes 6, 7, 8, 12, and 13 represent evacuation sources, while nodes 1, 20, and 22 are safe zones (sink nodes). The network's links have specified capacities and travel times.

The performance of the QAR algorithm was empirically compared to Dinic's algorithm, a state-of-the-art method known for its computational efficiency. A dynamic version of Dinic's algorithm was created using the time-expanded network technique. After five time intervals, Dinic's algorithm successfully evacuated 370 entities (vehicles) from sources to sinks within a total of 10 time intervals.

In contrast, the proposed time-dependent QAR algorithm, which actively adapts the road network's topology, demonstrated superior performance. After five time intervals, it evacuated 726 entities within the same 10-time interval period. These results highlight the effectiveness of the proposed method in outperforming existing state-of-the-art approaches for dynamic evacuation routing.

The code used for implementing, comparing, and evaluating the QAR algorithm against Dinic's algorithm is available in a GitHub repository.

**Usage**

QAR.py provides an illustrative application of the algorithm, generating a Manhattan-style road segment network with randomly assigned capacities and travel times for the road segments. The script utilizes a library called LibMaxFlow for its functions, and the resulting outputs are saved as graphical figures of the network.

**License**

Copyright (c) 2023 Hojat Behrooz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software") to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright and permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
