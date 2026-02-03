"""Compare Bellman-Ford and Dijkstra on a directed weighted graph."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

Vertex = int
Weight = float
Edge = Tuple[Vertex, Weight]
Graph = Dict[Vertex, List[Edge]]


@dataclass(frozen=True)
class ShortestPathResult:
    """Container for shortest-path results."""

    distances: Dict[Vertex, Weight]
    predecessors: Dict[Vertex, Optional[Vertex]]
    has_negative_cycle: bool = False


def iter_edges(graph: Graph) -> Iterable[Tuple[Vertex, Vertex, Weight]]:
    """Yield edges from an adjacency list as (u, v, weight)."""

    for u, neighbors in graph.items():
        for v, weight in neighbors:
            yield u, v, weight


def has_negative_weight(graph: Graph) -> bool:
    """Return True if any edge weight is negative."""

    return any(weight < 0 for _, _, weight in iter_edges(graph))


def bellman_ford(graph: Graph, source: Vertex) -> ShortestPathResult:
    """Compute shortest paths with Bellman-Ford and detect negative cycles."""

    distances = {vertex: float("inf") for vertex in graph}
    predecessors: Dict[Vertex, Optional[Vertex]] = {vertex: None for vertex in graph}
    distances[source] = 0.0

    for _ in range(len(graph) - 1):
        updated = False
        for u, v, weight in iter_edges(graph):
            if distances[u] != float("inf") and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
                updated = True
        if not updated:
            break

    has_negative_cycle = False
    for u, v, weight in iter_edges(graph):
        if distances[u] != float("inf") and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            break

    return ShortestPathResult(distances, predecessors, has_negative_cycle)


def dijkstra(graph: Graph, source: Vertex) -> ShortestPathResult:
    """Compute shortest paths with Dijkstra's algorithm."""

    distances = {vertex: float("inf") for vertex in graph}
    predecessors: Dict[Vertex, Optional[Vertex]] = {vertex: None for vertex in graph}
    distances[source] = 0.0

    heap: List[Tuple[Weight, Vertex]] = [(0.0, source)]
    visited: set[Vertex] = set()

    while heap:
        current_distance, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.get(u, []):
            new_distance = current_distance + weight
            if new_distance < distances[v]:
                distances[v] = new_distance
                predecessors[v] = u
                heapq.heappush(heap, (new_distance, v))

    return ShortestPathResult(distances, predecessors)


def measure_time(func, *args) -> Tuple[ShortestPathResult, float]:
    """Run a shortest-path function and return result plus elapsed seconds."""

    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def format_distances(distances: Dict[Vertex, Weight]) -> str:
    """Format distances for display."""

    lines = ["Shortest distances from the source:"]
    for vertex in sorted(distances):
        distance = distances[vertex]
        display = "∞" if distance == float("inf") else f"{distance:g}"
        lines.append(f"  {vertex}: {display}")
    return "\n".join(lines)


def compare_algorithms(bf_time: float, dj_time: float, negative_weights: bool) -> str:
    """Return a readable comparison of the two algorithms."""

    lines = [
        "Algorithm comparison:",
        f"  Bellman-Ford time: {bf_time:.6f}s (handles negative weights)",
        f"  Dijkstra time:     {dj_time:.6f}s "
        f"({'unsafe with negative weights' if negative_weights else 'requires non-negative weights'})",
        "  Efficiency: Dijkstra is typically faster on non-negative graphs,",
        "              while Bellman-Ford is more flexible but slower.",
    ]
    return "\n".join(lines)


def build_graph(num_vertices: int, edges: List[Tuple[Vertex, Vertex, Weight]]) -> Graph:
    """Construct an adjacency list graph from edge data."""

    graph: Graph = {vertex: [] for vertex in range(num_vertices)}
    for u, v, weight in edges:
        graph[u].append((v, weight))
    return graph


def prompt_for_graph() -> Tuple[Graph, Vertex]:
    """Prompt the user to input a graph interactively."""

    num_vertices = int(input("Number of vertices: ").strip())
    num_edges = int(input("Number of edges: ").strip())
    edges: List[Tuple[Vertex, Vertex, Weight]] = []

    print("Enter each edge as: <from> <to> <weight>")
    for _ in range(num_edges):
        u_str, v_str, w_str = input("Edge: ").split()
        edges.append((int(u_str), int(v_str), float(w_str)))

    source = int(input("Source vertex: ").strip())
    return build_graph(num_vertices, edges), source


def default_graph() -> Tuple[Graph, Vertex]:
    """Return the default sample graph and source."""

    graph: Graph = {
        0: [(1, 5), (2, 4)],
        1: [(3, 3)],
        2: [(1, 6), (3, 2)],
        3: [],
    }
    return graph, 0


def build_networkx_graph(graph: Graph) -> nx.DiGraph:
    """Create a NetworkX directed graph with weight attributes."""

    nx_graph = nx.DiGraph()
    for u, v, weight in iter_edges(graph):
        nx_graph.add_edge(u, v, weight=weight)
    return nx_graph


def shortest_path_edges(predecessors: Dict[Vertex, Optional[Vertex]]) -> set[Tuple[Vertex, Vertex]]:
    """Return the set of edges in the shortest-path tree."""

    edges = set()
    for vertex, predecessor in predecessors.items():
        if predecessor is not None:
            edges.add((predecessor, vertex))
    return edges


def visualize_paths(
    graph: Graph,
    bf_result: ShortestPathResult,
    dj_result: ShortestPathResult,
) -> None:
    """Visualize the graph and highlight shortest paths for each algorithm."""

    nx_graph = build_networkx_graph(graph)
    pos = nx.spring_layout(nx_graph, seed=42)

    bf_edges = shortest_path_edges(bf_result.predecessors)
    dj_edges = shortest_path_edges(dj_result.predecessors)

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(nx_graph, pos, node_size=700, node_color="#f0f0f0")
    nx.draw_networkx_labels(nx_graph, pos)

    base_edges = nx_graph.edges()
    base_colors = []
    for edge in base_edges:
        if edge in bf_edges and edge in dj_edges:
            base_colors.append("#6a0dad")  # purple for overlap
        elif edge in bf_edges:
            base_colors.append("#1f77b4")  # blue for Bellman-Ford
        elif edge in dj_edges:
            base_colors.append("#2ca02c")  # green for Dijkstra
        else:
            base_colors.append("#aaaaaa")

    nx.draw_networkx_edges(nx_graph, pos, edge_color=base_colors, arrows=True, width=2)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in nx_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels)

    plt.title("Shortest Path Trees (Blue: Bellman-Ford, Green: Dijkstra, Purple: Both)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run an interactive comparison of Bellman-Ford and Dijkstra."""

    use_custom = input("Provide a custom graph? (y/n): ").strip().lower()
    if use_custom == "y":
        graph, source = prompt_for_graph()
    else:
        graph, source = default_graph()

    if has_negative_weight(graph):
        print("Warning: graph contains negative weights.")
        print("Dijkstra's algorithm may produce incorrect results.")

    bf_result, bf_time = measure_time(bellman_ford, graph, source)
    dj_result, dj_time = measure_time(dijkstra, graph, source)

    print("\nBellman-Ford results")
    print(format_distances(bf_result.distances))
    if bf_result.has_negative_cycle:
        print("Warning: negative-weight cycle detected by Bellman-Ford.")

    print("\nDijkstra results")
    print(format_distances(dj_result.distances))

    print("\n" + compare_algorithms(bf_time, dj_time, has_negative_weight(graph)))

    visualize_paths(graph, bf_result, dj_result)


if __name__ == "__main__":
    main()
