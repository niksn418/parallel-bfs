# parallel-bfs

Total runtime for 5 runs, on cube graph with a side of length 400, running in 4 threads:
- seq: 74.347s
- par: 17.9628s
- Speedup: 4.13893x

**NOTE**: it is strange that speedup is greater than 4 while using only 4 threads. Probably it is due to some limitations of memory, as on another hardware with more RAM the speedup is only ~2.5x.