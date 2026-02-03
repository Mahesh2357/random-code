# random-code

## Shortest Path Comparison
## Bellman-Ford Example

Run the sample program:

```bash
python shortest_paths.py
```

Default graph:
python bellman_ford.py
```

Sample input:

```python
graph = {
    0: [(1, 5), (2, 4)],
    1: [(3, 3)],
    2: [(1, 6), (3, 2)],
    3: []
}
source = 0
```

Sample output:

```
Bellman-Ford results
Shortest distances from the source:
  0: 0
  1: 5
  2: 4
  3: 7

Dijkstra results
Shortest distances from the source:
  0: 0
  1: 5
  2: 4
  3: 7
```
