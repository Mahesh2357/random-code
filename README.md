# random-code

## Bellman-Ford Example

Run the sample program:

```bash
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
Shortest distances from the source:
  0: 0
  1: 5
  2: 4
  3: 7
```
