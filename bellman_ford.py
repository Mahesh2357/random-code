"""Bellman-Ford shortest paths with negative-cycle detection.

This module provides a readable, well-commented implementation of the
Bellman-Ford algorithm using an adjacency-list graph representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

Vertex = int
Weight = float
Edge = Tuple[Vertex, Weight]
Graph = Dict[Vertex, List[Edge]]


@dataclass(frozen=True)
class BellmanFordResult:
    """Result container for Bellman-Ford runs."""

    distances: Dict[Vertex, Weight]
    has_negative_cycle: bool


def _iter_edges(graph: Graph) -> Iterable[Tuple[Vertex, Vertex, Weight]]:
    """Yield edges as (u, v, weight) triples from an adjacency list."""

    for u, neighbors in graph.items():
        for v, weight in neighbors:
            yield u, v, weight


def bellman_ford(graph: Graph, source: Vertex) -> Dict[Vertex, Weight]:
    """Compute shortest path distances from a source vertex.

    Args:
        graph: Adjacency list representation of a directed, weighted graph.
        source: The starting vertex.

    Returns:
        A dictionary mapping each vertex to its shortest-path distance from the
        source. Unreachable vertices will have distance float("inf").
    """

    distances = {vertex: float("inf") for vertex in graph}
    distances[source] = 0.0

    # Relax all edges |V| - 1 times.
    for _ in range(len(graph) - 1):
        updated = False
        for u, v, weight in _iter_edges(graph):
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                updated = True
        if not updated:
            break

    return distances


def bellman_ford_with_cycle_check(
    graph: Graph, source: Vertex
) -> BellmanFordResult:
    """Run Bellman-Ford and detect negative-weight cycles.

    Args:
        graph: Adjacency list representation of a directed, weighted graph.
        source: The starting vertex.

    Returns:
        BellmanFordResult containing distances and a cycle flag.
    """

    distances = bellman_ford(graph, source)

    # One more pass to check for negative cycles.
    has_negative_cycle = False
    for u, v, weight in _iter_edges(graph):
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break

    return BellmanFordResult(distances=distances, has_negative_cycle=has_negative_cycle)


def _format_distances(distances: Dict[Vertex, Weight]) -> str:
    """Return a readable, multi-line string of distances."""

    lines = ["Shortest distances from the source:"]
    for vertex in sorted(distances):
        distance = distances[vertex]
        pretty_distance = "∞" if distance == float("inf") else f"{distance:g}"
        lines.append(f"  {vertex}: {pretty_distance}")
    return "\n".join(lines)


def main() -> None:
    """Run a sample Bellman-Ford computation."""

    # Sample graph (directed) using adjacency list representation.
    graph: Graph = {
        0: [(1, 5), (2, 4)],
        1: [(3, 3)],
        2: [(1, 6), (3, 2)],
        3: [],
    }
    source = 0

    result = bellman_ford_with_cycle_check(graph, source)
    print(_format_distances(result.distances))

    if result.has_negative_cycle:
        print("Warning: the graph contains a negative-weight cycle.")


if __name__ == "__main__":
    main()
