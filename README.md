# PMA
### Scalable parallel minimum spanning forest computation

## Abstract

The proliferation of data in graph form calls for the development of scalable graph algorithms that exploit parallel processing environments. One such problem is the computation of a graph's minimum spanning forest (MSF). Past research has proposed several parallel algorithms for this problem, yet none of them scales to large, high-density graphs. In this paper we propose a novel, scalable, Parallel MSF Algorithm (PMA) for undirected weighted graphs. Our algorithm leverages Prim's algorithm in a parallel fashion, concurrently expanding several subsets of the computed MSF. Our effort focuses on minimizing the communication among different processors without constraining the local growth of a processor's computed subtree. In effect, we achieve a scalability that previous approaches lacked. We implement our algorithm in CUDA, running on a GPU and study its performance using real and synthetic, sparse as well as dense, structured and unstructured graph data. Our experimental study demonstrates that our algorithm outperforms the previous state-of-the-art GPU-based MSF algorithm, while being several orders of magnitude faster than sequential CPU-based algorithms.

## Please cite the paper:
[Sadegh Nobari](http://bit.ly/NOB-GS), Thanh-Tung Cao, Panagiotis Karras, and Stéphane Bressan,
["Scalable parallel minimum spanning forest computation"](http://dl.acm.org/citation.cfm?id=2145842),
Proc. of the 17th ACM SIGPLAN symposium on Principles and Practice of Parallel Programming (PPoPP'12), New Orleans, LA, USA

## Furthur reading
[Full paper](http://dl.acm.org/citation.cfm?id=2145842)

[Wikipedia](http://en.wikipedia.org/wiki/Minimum_spanning_tree)
