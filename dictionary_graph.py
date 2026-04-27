#!/usr/bin/env python3
"""
Dictionary Graph Explorer
=========================
Builds a directed graph from dictionary definitions and analyzes:
  a) Connected components (islands) and their sizes
  b) Degrees of separation from a starting word
  c) Island fragmentation as BFS expands
  d) Shortest definitional loops (cycles)
  e) Longest definitional loops (cycles) — exact for small SCCs, heuristic for large

Supports two dictionary backends:
  --dict webster   Webster's Unabridged 1913 (public domain JSON from GitHub)
  --dict wordnet   WordNet 3.0 via NLTK

Usage:
  python dictionary_graph.py --dict webster          # full analysis, random start
  python dictionary_graph.py --dict wordnet --start dog
  python dictionary_graph.py --dict webster --download-only
"""

import argparse
import io
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEBSTER_URLS = [
    "https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary.json",
    "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json",
]
WEBSTER_CACHE = "webster_1913.json"
RESULTS_FILE = "results.txt"

_WORD_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*")

# ---------------------------------------------------------------------------
# Utility: tee print to console + file
# ---------------------------------------------------------------------------

_results_file_handle = None


def tee(msg: str = ""):
    """Print to console and optionally to results file."""
    print(msg)
    if _results_file_handle:
        _results_file_handle.write(msg + "\n")
        _results_file_handle.flush()


def fmt_pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{100.0 * part / whole:.1f}%"


def fmt_num(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Dictionary loading — Webster's 1913
# ---------------------------------------------------------------------------

def download_webster(cache_path: str = WEBSTER_CACHE) -> dict:
    """Download Webster's 1913 dictionary JSON, caching locally."""
    if os.path.exists(cache_path):
        print(f"  Loading cached Webster's from {cache_path} ...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for url in WEBSTER_URLS:
        try:
            print(f"  Downloading Webster's 1913 from:\n    {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "DictionaryGraph/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(cache_path, "wb") as f:
                f.write(data)
            print(f"  Saved {len(data):,} bytes → {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  ✗ Failed ({e}), trying next URL...")

    print("ERROR: Could not download Webster's dictionary from any source.")
    sys.exit(1)


def load_webster(cache_path: str = WEBSTER_CACHE) -> dict[str, str]:
    """Load Webster's 1913. Returns {word: definition}."""
    raw = download_webster(cache_path)
    result: dict[str, str] = {}
    for word, defn in raw.items():
        key = word.strip().lower()
        if not key or not defn or not key[0].isalpha():
            continue
        # Some entries are lists or non-string; coerce
        if isinstance(defn, list):
            defn = " ".join(str(d) for d in defn)
        elif not isinstance(defn, str):
            defn = str(defn)
        if key in result:
            result[key] += " " + defn
        else:
            result[key] = defn
    return result


# ---------------------------------------------------------------------------
# Dictionary loading — WordNet (NLTK)
# ---------------------------------------------------------------------------

def load_wordnet() -> dict[str, str]:
    """Load WordNet via NLTK. Returns {word: combined_definition}."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("ERROR: NLTK not installed.  Run:  pip install nltk")
        sys.exit(1)

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    result: dict[str, str] = {}
    count = 0
    for synset in wn.all_synsets():
        defn = synset.definition()
        if not defn:
            continue
        for lemma_name in synset.lemma_names():
            word = lemma_name.replace("_", " ").lower().strip()
            if not word or not word[0].isalpha():
                continue
            # For graph purposes, also add the single-word version
            # (multi-word entries like "ice cream" are kept as-is)
            if word in result:
                result[word] += "; " + defn
            else:
                result[word] = defn
        count += 1
        if count % 20000 == 0:
            print(f"  Loaded {count:,} synsets...")

    print(f"  Loaded {count:,} synsets → {len(result):,} unique lemmas")
    return result


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def tokenize_definition(definition: str) -> set[str]:
    """Extract unique lowercase words from a definition string."""
    return set(_WORD_RE.findall(definition.lower()))


def build_graph(
    dictionary: dict[str, str],
) -> tuple[dict[str, set[str]], set[str], dict[str, int]]:
    """
    Build directed graph: word → {words in its definition that are also in the dict}.

    Returns:
        graph:       adjacency dict (no self-loops)
        self_refs:   set of words whose definition contains themselves
        def_lengths: word → number of in-vocab tokens in its definition
    """
    vocab = set(dictionary.keys())
    graph: dict[str, set[str]] = {}
    self_refs: set[str] = set()
    def_lengths: dict[str, int] = {}

    total = len(dictionary)
    report_every = max(1, total // 20)

    for i, (word, defn) in enumerate(dictionary.items()):
        if i % report_every == 0:
            pct = 100 * i // total
            print(f"\r  Building graph: {i:,}/{total:,} ({pct}%)", end="", flush=True)

        tokens = tokenize_definition(defn)
        neighbors = tokens & vocab
        if word in neighbors:
            self_refs.add(word)
            neighbors.discard(word)
        graph[word] = neighbors
        def_lengths[word] = len(neighbors)

    print(f"\r  Building graph: {total:,}/{total:,} (100%)    ")

    total_edges = sum(len(v) for v in graph.values())
    print(f"  Nodes: {len(graph):,}   Edges: {total_edges:,}   Self-refs: {len(self_refs):,}")
    return graph, self_refs, def_lengths


# ---------------------------------------------------------------------------
# Connected components (undirected view)
# ---------------------------------------------------------------------------

def find_connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    """Find connected components treating the graph as undirected."""
    # Build undirected adjacency (only for nodes in graph)
    all_words = set(graph.keys())
    undirected: dict[str, set[str]] = defaultdict(set)

    for word, neighbors in graph.items():
        for n in neighbors:
            if n in all_words:  # only consider edges within graph nodes
                undirected[word].add(n)
                undirected[n].add(word)

    visited: set[str] = set()
    components: list[set[str]] = []

    for word in all_words:
        if word in visited:
            continue
        component: set[str] = set()
        queue = deque([word])
        while queue:
            w = queue.popleft()
            if w in visited:
                continue
            visited.add(w)
            component.add(w)
            for neighbor in undirected.get(w, ()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    return sorted(components, key=len, reverse=True)


# ---------------------------------------------------------------------------
# BFS degrees of separation + island fragmentation
# ---------------------------------------------------------------------------

def bfs_degrees_and_islands(
    graph: dict[str, set[str]], start: str
) -> tuple[list[tuple[int, int, int]], int]:
    """
    BFS from `start` word, following directed definition links.

    Returns:
        depths: list of (depth, new_words_at_depth, cumulative_words)
        max_depth: depth at which all reachable words were found
    """
    visited = {start}
    frontier = {start}
    depths: list[tuple[int, int, int]] = [(0, 1, 1)]
    depth = 0

    while frontier:
        depth += 1
        next_frontier: set[str] = set()
        for word in frontier:
            for neighbor in graph.get(word, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        cumulative = len(visited)
        depths.append((depth, len(next_frontier), cumulative))
        frontier = next_frontier

    return depths, depth


def compute_island_fragmentation(
    graph: dict[str, set[str]], start: str
) -> list[tuple[int, int, list[int]]]:
    """
    At each BFS depth from `start`, compute connected components among
    the NOT-YET-REACHED words.

    Returns list of (depth, num_unreached_components, top_5_component_sizes).
    """
    all_words = set(graph.keys())

    # Build undirected adjacency once
    undirected: dict[str, set[str]] = defaultdict(set)
    for word, neighbors in graph.items():
        for n in neighbors:
            if n in all_words:
                undirected[word].add(n)
                undirected[n].add(word)

    visited = {start}
    frontier = {start}
    results: list[tuple[int, int, list[int]]] = []

    depth = 0

    def count_components_of(remaining: set[str]) -> tuple[int, list[int]]:
        """Count connected components among `remaining` words."""
        seen: set[str] = set()
        sizes: list[int] = []
        for w in remaining:
            if w in seen:
                continue
            size = 0
            stack = [w]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                size += 1
                for nb in undirected.get(node, ()):
                    if nb in remaining and nb not in seen:
                        stack.append(nb)
            sizes.append(size)
        sizes.sort(reverse=True)
        return len(sizes), sizes[:5]

    remaining = all_words - visited
    n_comp, top_sizes = count_components_of(remaining)
    results.append((0, n_comp, top_sizes))

    while frontier:
        depth += 1
        next_frontier: set[str] = set()
        for word in frontier:
            for neighbor in graph.get(word, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
        remaining = all_words - visited
        if not remaining:
            results.append((depth, 0, []))
            break
        n_comp, top_sizes = count_components_of(remaining)
        results.append((depth, n_comp, top_sizes))

    return results


# ---------------------------------------------------------------------------
# Undirected BFS — true "degrees of separation"
# ---------------------------------------------------------------------------

def bfs_undirected(
    graph: dict[str, set[str]], start: str
) -> tuple[list[tuple[int, int, int]], int]:
    """
    BFS from `start` over the UNDIRECTED version of the graph.
    A ↔ B if A's definition mentions B  OR  B's definition mentions A.

    Returns same shape as bfs_degrees_and_islands.
    """
    all_words = set(graph.keys())
    undirected: dict[str, set[str]] = defaultdict(set)
    for word, neighbors in graph.items():
        for n in neighbors:
            if n in all_words:
                undirected[word].add(n)
                undirected[n].add(word)

    visited = {start}
    frontier = {start}
    depths: list[tuple[int, int, int]] = [(0, 1, 1)]
    depth = 0

    while frontier:
        depth += 1
        next_frontier: set[str] = set()
        for word in frontier:
            for neighbor in undirected.get(word, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        depths.append((depth, len(next_frontier), len(visited)))
        frontier = next_frontier

    return depths, depth


# ---------------------------------------------------------------------------
# Count ALL 2-cycles
# ---------------------------------------------------------------------------

def count_all_2cycles(graph: dict[str, set[str]]) -> tuple[int, list[tuple[str, str]]]:
    """Count every mutual-definition pair.  Returns (count, first_50_examples)."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for u, neighbors in graph.items():
        for v in neighbors:
            pair = (min(u, v), max(u, v))
            if pair not in seen and u in graph.get(v, ()):
                pairs.append(pair)
                seen.add(pair)
    examples = [(a, b) for a, b in pairs[:50]]
    return len(pairs), examples


# ---------------------------------------------------------------------------
# Strongly Connected Components — iterative Kosaraju's
# ---------------------------------------------------------------------------

def kosaraju_scc(graph: dict[str, set[str]]) -> list[set[str]]:
    """Iterative Kosaraju's algorithm for Strongly Connected Components."""
    all_nodes = set(graph.keys())

    # Phase 1: iterative DFS on original graph — record finish order
    visited: set[str] = set()
    finish_order: list[str] = []

    for start in all_nodes:
        if start in visited:
            continue
        # Iterative DFS using explicit stack
        # Stack entries: (node, iterator_over_neighbors, is_entered)
        stack: list[tuple[str, ...]] = [(start,)]
        while stack:
            frame = stack[-1]
            node = frame[0]
            if len(frame) == 1:
                # First visit
                if node in visited:
                    stack.pop()
                    continue
                visited.add(node)
                neighbors = list(graph.get(node, ()))
                stack[-1] = (node, 0, len(neighbors), neighbors)
                continue
            # Resuming: try next unvisited neighbor
            _, idx, length, neighbors = frame  # type: ignore
            pushed = False
            while idx < length:
                nb = neighbors[idx]
                idx += 1
                if nb not in visited:
                    stack[-1] = (node, idx, length, neighbors)
                    stack.append((nb,))
                    pushed = True
                    break
            if not pushed:
                # All neighbors explored — record finish
                finish_order.append(node)
                stack.pop()

    # Phase 2: build transposed graph
    transposed: dict[str, list[str]] = defaultdict(list)
    for node, neighbors in graph.items():
        for n in neighbors:
            transposed[n].append(node)

    # Phase 3: DFS on transposed graph in reverse finish order
    visited2: set[str] = set()
    sccs: list[set[str]] = []

    for node in reversed(finish_order):
        if node in visited2:
            continue
        component: set[str] = set()
        dfs_stack = [node]
        while dfs_stack:
            n = dfs_stack.pop()
            if n in visited2:
                continue
            visited2.add(n)
            component.add(n)
            for nb in transposed.get(n, ()):
                if nb not in visited2:
                    dfs_stack.append(nb)
        sccs.append(component)

    return sccs


# ---------------------------------------------------------------------------
# Shortest cycle finding
# ---------------------------------------------------------------------------

def find_shortest_cycles(
    graph: dict[str, set[str]], sccs: list[set[str]], max_examples: int = 20
) -> tuple[int, list[list[str]]]:
    """
    Find the shortest directed cycle(s) in the graph.

    Strategy:
      1. Check for 2-cycles (A→B→A) — O(E)
      2. If none, BFS from each node in non-trivial SCCs — O(V*(V+E)) worst case
         but with early termination once min_length is found.

    Returns (cycle_length, list_of_example_cycles_as_word_lists).
    """
    # ---------- 2-cycles ----------
    two_cycles: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for u, neighbors in graph.items():
        for v in neighbors:
            pair = (min(u, v), max(u, v))
            if pair in seen_pairs:
                continue
            if u in graph.get(v, ()):
                two_cycles.append([u, v, u])
                seen_pairs.add(pair)
                if len(two_cycles) >= max_examples * 10:
                    break
        if len(two_cycles) >= max_examples * 10:
            break

    if two_cycles:
        return 2, two_cycles[:max_examples]

    # ---------- 3-cycles ----------
    tee("  No 2-cycles found; scanning for 3-cycles...")
    three_cycles: list[list[str]] = []
    seen_triples: set[tuple[str, ...]] = set()

    for u, u_neighbors in graph.items():
        for v in u_neighbors:
            for w in graph.get(v, ()):
                if w == u:
                    continue
                if u in graph.get(w, ()):
                    triple = tuple(sorted([u, v, w]))
                    if triple not in seen_triples:
                        three_cycles.append([u, v, w, u])
                        seen_triples.add(triple)
                        if len(three_cycles) >= max_examples * 10:
                            break
            if len(three_cycles) >= max_examples * 10:
                break
        if len(three_cycles) >= max_examples * 10:
            break

    if three_cycles:
        return 3, three_cycles[:max_examples]

    # ---------- General BFS for cycles of length 4+ ----------
    tee("  No 3-cycles found; BFS scanning for shortest cycle...")
    min_length = float("inf")
    best_cycles: list[list[str]] = []

    # Only search within non-trivial SCCs
    nontrivial_nodes: set[str] = set()
    node_to_scc: dict[str, int] = {}
    for i, scc in enumerate(sccs):
        if len(scc) >= 2:
            nontrivial_nodes.update(scc)
            for w in scc:
                node_to_scc[w] = i

    if not nontrivial_nodes:
        return 0, []

    checked = 0
    for start in nontrivial_nodes:
        checked += 1
        if checked % 5000 == 0:
            tee(f"    BFS cycle search: checked {checked:,} / {len(nontrivial_nodes):,} nodes, "
                f"best so far: {min_length}")

        scc_id = node_to_scc[start]
        scc_set = sccs[scc_id]

        # BFS from start's neighbors, looking for path back to start
        # (predecessor map for path reconstruction)
        parent: dict[str, Optional[str]] = {}
        dist: dict[str, int] = {}
        queue: deque[str] = deque()

        for nb in graph.get(start, ()):
            if nb in scc_set and nb != start:
                dist[nb] = 1
                parent[nb] = start
                queue.append(nb)

        found = False
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d + 1 >= min_length:
                break  # can't improve
            for nb in graph.get(node, ()):
                if nb == start:
                    cycle_len = d + 1
                    if cycle_len < min_length:
                        # Reconstruct path
                        path = [start]
                        cur = node
                        while cur != start:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(start)
                        min_length = cycle_len
                        best_cycles = [path]
                    elif cycle_len == min_length:
                        path = [start]
                        cur = node
                        while cur != start:
                            path.append(cur)
                            cur = parent[cur]
                        path.append(start)
                        if len(best_cycles) < max_examples:
                            best_cycles.append(path)
                    found = True
                    break
                if nb in scc_set and nb not in dist:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
            if found:
                break

    return int(min_length) if min_length != float("inf") else 0, best_cycles


# ---------------------------------------------------------------------------
# Longest cycle finding (heuristic for large SCCs, exact for small)
# ---------------------------------------------------------------------------

def find_longest_cycle_in_small_scc(
    graph: dict[str, set[str]], scc: set[str]
) -> tuple[int, list[str]]:
    """
    For SCCs with ≤ 25 nodes, find the longest simple cycle exactly via
    DFS with backtracking.
    """
    nodes = list(scc)
    subgraph: dict[str, list[str]] = {}
    for n in nodes:
        subgraph[n] = [nb for nb in graph.get(n, ()) if nb in scc]

    best_len = 0
    best_path: list[str] = []

    for start in nodes:
        # DFS backtracking
        stack: list[tuple[str, list[str], set[str]]] = [
            (start, [start], {start})
        ]
        while stack:
            node, path, visited = stack.pop()
            for nb in subgraph.get(node, ()):
                if nb == start and len(path) > 1:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = path + [start]
                elif nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))

    return best_len, best_path


def find_longest_cycle_heuristic(
    graph: dict[str, set[str]], scc: set[str], time_limit: float = 30.0
) -> tuple[int, list[str]]:
    """
    Heuristic search for long cycles in large SCCs.

    Strategy: random walk forward for a bounded number of steps, then
    BFS back to the start through un-walked nodes.  We try many walk
    lengths and start nodes, keeping the longest cycle found.
    """
    nodes = list(scc)
    scc_set = scc  # for fast membership
    n_scc = len(nodes)
    subgraph: dict[str, list[str]] = {}
    for n in nodes:
        subgraph[n] = [nb for nb in graph.get(n, ()) if nb in scc_set]

    best_len = 0
    best_path: list[str] = []
    t0 = time.time()
    attempts = 0

    # Try progressively longer walk caps
    walk_caps = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000,
                 n_scc // 4, n_scc // 2]
    walk_caps = sorted(set(c for c in walk_caps if 2 <= c <= n_scc))

    cap_idx = 0

    while time.time() - t0 < time_limit:
        attempts += 1
        max_walk = walk_caps[cap_idx % len(walk_caps)]
        cap_idx += 1

        # --- Forward walk (capped at max_walk steps) ---
        start = random.choice(nodes)
        path = [start]
        visited = {start}
        current = start

        use_greedy = (attempts % 3 != 0)

        for _ in range(max_walk):
            neighbors = [nb for nb in subgraph.get(current, ()) if nb not in visited]
            if not neighbors:
                break
            if use_greedy and len(neighbors) <= 50:
                current = max(
                    neighbors,
                    key=lambda nb: sum(
                        1 for x in subgraph.get(nb, ()) if x not in visited
                    ),
                )
            else:
                current = random.choice(neighbors)
            path.append(current)
            visited.add(current)

        if len(path) < 2:
            continue

        # --- BFS return: shortest path from last → start, avoiding walked nodes ---
        last = path[-1]
        forbidden = visited - {start}  # start is the goal, allow it
        parent: dict[str, str] = {}
        queue: deque[str] = deque()
        found_cycle = False

        # Seed BFS with last's neighbors
        for nb in subgraph.get(last, ()):
            if nb == start:
                # Direct return edge
                cycle_len = len(path)
                if cycle_len > best_len:
                    best_len = cycle_len
                    best_path = path + [start]
                found_cycle = True
                break
            if nb not in forbidden and nb in scc_set and nb not in parent:
                parent[nb] = last
                queue.append(nb)

        if not found_cycle:
            while queue:
                node = queue.popleft()
                for nb in subgraph.get(node, ()):
                    if nb == start:
                        # Reconstruct return path: last → ... → node → start
                        return_nodes: list[str] = []
                        cur = node
                        while cur != last:
                            return_nodes.append(cur)
                            cur = parent[cur]
                        return_nodes.reverse()
                        # full cycle: path + return_nodes + [start]
                        full_cycle = path + return_nodes + [start]
                        cycle_len = len(path) + len(return_nodes)
                        if cycle_len > best_len:
                            best_len = cycle_len
                            best_path = full_cycle
                        found_cycle = True
                        break
                    if nb not in forbidden and nb in scc_set and nb not in parent:
                        parent[nb] = node
                        queue.append(nb)
                if found_cycle:
                    break

        if attempts % 20 == 0:
            elapsed = time.time() - t0
            tee(f"    ... {attempts} attempts in {elapsed:.1f}s, "
                f"best cycle: {best_len}, walk cap: {max_walk}")

    tee(f"    Done: {attempts} attempts in {time.time()-t0:.1f}s, best cycle: {best_len}")
    return best_len, best_path


def find_longest_cycles(
    graph: dict[str, set[str]],
    sccs: list[set[str]],
    time_limit: float = 60.0,
) -> list[tuple[int, list[str], str]]:
    """
    Find longest cycles across all SCCs.

    Returns list of (cycle_length, cycle_path, method_note).
    """
    results: list[tuple[int, list[str], str]] = []
    nontrivial = [scc for scc in sccs if len(scc) >= 2]
    nontrivial.sort(key=len, reverse=True)

    for i, scc in enumerate(nontrivial):
        size = len(scc)
        if size == 2:
            # Only possible cycle is length 2
            nodes = list(scc)
            a, b = nodes
            if b in graph.get(a, ()) and a in graph.get(b, ()):
                results.append((2, [a, b, a], "exact (size-2 SCC)"))
            continue

        if size <= 25:
            tee(f"  SCC #{i+1} (size {size}): exact search...")
            length, path = find_longest_cycle_in_small_scc(graph, scc)
            if length > 0:
                method = f"exact (exhaustive, SCC size {size})"
                results.append((length, path, method))
        else:
            # Time budget proportional to SCC size (but capped)
            budget = min(time_limit, max(5.0, time_limit * size / len(nontrivial[0])))
            tee(f"  SCC #{i+1} (size {size:,}): heuristic search ({budget:.0f}s budget)...")
            length, path = find_longest_cycle_heuristic(graph, scc, time_limit=budget)
            if length > 0:
                method = f"heuristic (SCC size {size:,}, upper bound = {size:,})"
                results.append((length, path, method))

        # Only do detailed search on top 10 non-trivial SCCs
        if i >= 9:
            remaining = len(nontrivial) - i - 1
            if remaining > 0:
                tee(f"  (skipping {remaining} smaller SCCs for longest-cycle search)")
            break

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    dict_name: str,
    dictionary: dict[str, str],
    start_word: Optional[str],
    cycle_time_limit: float,
    skip_fragmentation: bool,
):
    global _results_file_handle

    results_path = f"results_{dict_name}.txt"
    _results_file_handle = open(results_path, "w", encoding="utf-8")

    tee("=" * 72)
    tee(f"  DICTIONARY GRAPH ANALYSIS — {dict_name.upper()}")
    tee("=" * 72)
    tee()

    # ---- Build graph ----
    tee("─── Building Graph ───")
    t0 = time.time()
    graph, self_refs, def_lengths = build_graph(dictionary)
    build_time = time.time() - t0
    total_words = len(graph)
    total_edges = sum(len(v) for v in graph.values())
    avg_edges = total_edges / total_words if total_words else 0

    tee(f"  Dictionary:          {dict_name}")
    tee(f"  Total words (nodes): {fmt_num(total_words)}")
    tee(f"  Total edges:         {fmt_num(total_edges)}")
    tee(f"  Avg edges per word:  {avg_edges:.1f}")
    tee(f"  Self-referential:    {fmt_num(len(self_refs))} words define themselves")
    tee(f"  Build time:          {build_time:.1f}s")

    if self_refs:
        examples = sorted(self_refs)[:15]
        tee(f"  Self-ref examples:   {', '.join(examples)}")
    tee()

    # ---- Choose start word ----
    all_words = list(graph.keys())
    if start_word:
        if start_word.lower() not in graph:
            tee(f"  WARNING: '{start_word}' not in dictionary; picking random word.")
            start_word = random.choice(all_words)
        else:
            start_word = start_word.lower()
    else:
        start_word = random.choice(all_words)
    tee(f"  Starting word:       \"{start_word}\"")
    tee(f"  Its definition:      {dictionary.get(start_word, '???')[:200]}")
    tee()

    # ---- Connected components (undirected) ----
    tee("─── Connected Components (undirected graph) ───")
    t0 = time.time()
    components = find_connected_components(graph)
    cc_time = time.time() - t0
    tee(f"  Total components:    {fmt_num(len(components))}")
    tee(f"  Computation time:    {cc_time:.1f}s")
    tee()

    # Size distribution
    tee("  Rank │   Size   │  % of total │ Sample words")
    tee("  ─────┼──────────┼─────────────┼─────────────────────────────")
    for i, comp in enumerate(components[:20]):
        sample = sorted(comp)[:5]
        sample_str = ", ".join(sample)
        if len(comp) > 5:
            sample_str += ", ..."
        tee(f"  {i+1:4d} │ {len(comp):>8,} │ {fmt_pct(len(comp), total_words):>11} │ {sample_str}")
    if len(components) > 20:
        tee(f"  ... and {len(components) - 20:,} more components")

    # Size histogram
    size_counts: dict[int, int] = defaultdict(int)
    for comp in components:
        size_counts[len(comp)] += 1
    tee()
    tee("  Component size histogram:")
    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        tee(f"    Size {size:>8,}: {count:>6,} component(s)")
    tee()

    # ---- BFS degrees of separation (DIRECTED) ----
    tee(f'─── Directed BFS from "{start_word}" ───')
    tee(f'  (follow definition links outward: A→B means B appears in A\'s definition)')
    t0 = time.time()
    depths, max_depth = bfs_degrees_and_islands(graph, start_word)
    bfs_time = time.time() - t0

    tee(f"  Max depth to reach all reachable words: {max_depth}")
    tee(f"  BFS time: {bfs_time:.1f}s")
    tee()
    tee("  Depth │    New words │   Cumulative │  % of total")
    tee("  ──────┼──────────────┼──────────────┼─────────────")
    for depth, new_count, cum_count in depths:
        tee(f"  {depth:5d} │ {new_count:>12,} │ {cum_count:>12,} │ {fmt_pct(cum_count, total_words):>11}")

    reachable_dir = depths[-1][2] if depths else 0
    unreachable_dir = total_words - reachable_dir
    if unreachable_dir > 0:
        tee(f"\n  ⚠ {fmt_num(unreachable_dir)} words ({fmt_pct(unreachable_dir, total_words)}) "
            f"unreachable from \"{start_word}\" via directed edges")
    tee()

    # ---- BFS degrees of separation (UNDIRECTED) ----
    tee(f'─── Undirected BFS from "{start_word}" (true degrees of separation) ───')
    tee(f'  (A↔B if A\'s definition mentions B  OR  B\'s definition mentions A)')
    t0 = time.time()
    u_depths, u_max_depth = bfs_undirected(graph, start_word)
    u_bfs_time = time.time() - t0

    tee(f"  Max depth to reach all reachable words: {u_max_depth}")
    tee(f"  BFS time: {u_bfs_time:.1f}s")
    tee()
    tee("  Depth │    New words │   Cumulative │  % of total")
    tee("  ──────┼──────────────┼──────────────┼─────────────")
    for depth, new_count, cum_count in u_depths:
        tee(f"  {depth:5d} │ {new_count:>12,} │ {cum_count:>12,} │ {fmt_pct(cum_count, total_words):>11}")

    reachable_undir = u_depths[-1][2] if u_depths else 0
    unreachable_undir = total_words - reachable_undir
    if unreachable_undir > 0:
        tee(f"\n  ⚠ {fmt_num(unreachable_undir)} words ({fmt_pct(unreachable_undir, total_words)}) "
            f"unreachable from \"{start_word}\" even in undirected graph")
    else:
        tee(f"\n  ✓ ALL words in the main component reachable within {u_max_depth} hops")
    tee()

    # ---- Island fragmentation ----
    if not skip_fragmentation:
        tee(f'─── Island Fragmentation (unexplored territory at each directed BFS depth) ───')
        t0 = time.time()
        fragmentation = compute_island_fragmentation(graph, start_word)
        frag_time = time.time() - t0
        tee(f"  Computation time: {frag_time:.1f}s")
        tee()
        tee("  Depth │ Unreached islands │ Largest island sizes")
        tee("  ──────┼───────────────────┼────────────────────────────────")
        for depth, n_comp, top_sizes in fragmentation:
            sizes_str = ", ".join(f"{s:,}" for s in top_sizes)
            tee(f"  {depth:5d} │ {n_comp:>17,} │ {sizes_str}")
        tee()
    else:
        tee("  (fragmentation analysis skipped — use --fragmentation to enable)")
        tee()

    # ---- Strongly Connected Components ----
    tee("─── Strongly Connected Components (directed graph) ───")
    t0 = time.time()
    sccs = kosaraju_scc(graph)
    scc_time = time.time() - t0

    nontrivial_sccs = [s for s in sccs if len(s) >= 2]
    tee(f"  Total SCCs:           {fmt_num(len(sccs))}")
    tee(f"  Non-trivial (size≥2): {fmt_num(len(nontrivial_sccs))}")
    tee(f"  Computation time:     {scc_time:.1f}s")
    tee()

    scc_sizes = sorted([len(s) for s in sccs], reverse=True)
    scc_size_dist: dict[int, int] = defaultdict(int)
    for s in scc_sizes:
        scc_size_dist[s] += 1

    tee("  SCC size distribution:")
    for size in sorted(scc_size_dist.keys(), reverse=True)[:20]:
        count = scc_size_dist[size]
        tee(f"    Size {size:>8,}: {count:>6,} SCC(s)")
    if len(scc_size_dist) > 20:
        tee(f"    ... and {len(scc_size_dist) - 20} more size classes")
    tee()

    # Top SCCs
    sccs_sorted = sorted(sccs, key=len, reverse=True)
    tee("  Top 10 SCCs:")
    for i, scc in enumerate(sccs_sorted[:10]):
        sample = sorted(scc)[:8]
        sample_str = ", ".join(sample)
        if len(scc) > 8:
            sample_str += ", ..."
        tee(f"    #{i+1}: {len(scc):>8,} words — {sample_str}")
    tee()

    # ---- Shortest cycles + full 2-cycle census ----
    tee("─── Shortest Definitional Loops ───")
    t0 = time.time()
    shortest_len, shortest_examples = find_shortest_cycles(graph, sccs)
    short_time = time.time() - t0

    if shortest_len == 0:
        tee("  No cycles found (all SCCs are singletons).")
    else:
        tee(f"  Shortest cycle length: {shortest_len}")
        tee(f"  Search time:           {short_time:.1f}s")
        tee()
        for i, cycle in enumerate(shortest_examples[:15]):
            arrow_path = " → ".join(cycle)
            tee(f"    {i+1:3d}. {arrow_path}")
        if len(shortest_examples) > 15:
            tee(f"    ... and {len(shortest_examples) - 15} more shown")

    tee()
    tee("  Counting ALL mutual-definition pairs (2-cycles)...")
    t0 = time.time()
    n_2cycles, two_cycle_examples = count_all_2cycles(graph)
    tee(f"  Total 2-cycles in dictionary: {fmt_num(n_2cycles)}")
    tee(f"  Count time: {time.time() - t0:.1f}s")
    if two_cycle_examples:
        tee("  Sample pairs:")
        for a, b in two_cycle_examples[:20]:
            tee(f"    {a} <-> {b}")
        if n_2cycles > 20:
            tee(f"    ... and {n_2cycles - 20:,} more")
    tee()

    # ---- Longest cycles ----
    tee(f"─── Longest Definitional Loops (time budget: {cycle_time_limit:.0f}s) ───")
    t0 = time.time()
    longest_results = find_longest_cycles(graph, sccs, time_limit=cycle_time_limit)
    long_time = time.time() - t0

    if not longest_results:
        tee("  No cycles found.")
    else:
        tee(f"  Search time: {long_time:.1f}s")
        tee()
        for i, (length, path, method) in enumerate(longest_results[:10]):
            tee(f"  #{i+1}: length {length:,}  ({method})")
            # Print path (abbreviated if long)
            if length <= 30:
                tee(f"       {' → '.join(path)}")
            else:
                head = " → ".join(path[:10])
                tail = " → ".join(path[-5:])
                tee(f"       {head}")
                tee(f"         ... ({length - 15} more) ...")
                tee(f"       {tail}")
            tee()
    tee()

    # ---- Summary statistics ----
    tee("=" * 72)
    tee("  SUMMARY")
    tee("=" * 72)
    tee(f"  Dictionary:                {dict_name}")
    tee(f"  Total words:               {fmt_num(total_words)}")
    tee(f"  Total directed edges:      {fmt_num(total_edges)}")
    tee(f"  Avg edges per word:        {avg_edges:.1f}")
    tee()
    tee(f"  --- Connectivity ---")
    biggest_cc = len(components[0]) if components else 0
    tee(f"  Connected components:      {fmt_num(len(components))}")
    tee(f"  Largest component:         {fmt_num(biggest_cc)} ({fmt_pct(biggest_cc, total_words)})")
    tee(f"  Isolated islands:          {len(components) - 1}")
    biggest_scc = scc_sizes[0] if scc_sizes else 0
    tee(f"  Strongly connected comps:  {fmt_num(len(sccs))}")
    tee(f"  Largest SCC:               {fmt_num(biggest_scc)} ({fmt_pct(biggest_scc, total_words)})")
    tee()
    tee(f"  --- Degrees of Separation (from \"{start_word}\") ---")
    tee(f"  Directed BFS max depth:    {max_depth} (reached {fmt_pct(reachable_dir, total_words)})")
    tee(f"  Undirected BFS max depth:  {u_max_depth} (reached {fmt_pct(reachable_undir, total_words)})")
    tee()
    tee(f"  --- Cycles ---")
    tee(f"  Self-referential words:    {fmt_num(len(self_refs))} ({fmt_pct(len(self_refs), total_words)})")
    tee(f"  Total 2-cycles (pairs):    {fmt_num(n_2cycles)}")
    tee(f"  Shortest cycle:            {shortest_len if shortest_len else 'none'}")
    if longest_results:
        tee(f"  Longest cycle found:       {longest_results[0][0]:,} ({longest_results[0][2]})")
    else:
        tee(f"  Longest cycle found:       none")
    tee()
    tee(f"  Results written to: {results_path}")
    tee("=" * 72)

    _results_file_handle.close()
    _results_file_handle = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dictionary Graph Explorer — analyze definitional loops and connectivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dictionary_graph.py --dict webster
  python dictionary_graph.py --dict wordnet --start dog
  python dictionary_graph.py --dict webster --download-only
  python dictionary_graph.py --dict webster --cycle-time 120 --fragmentation
        """,
    )
    parser.add_argument(
        "--dict",
        choices=["webster", "wordnet"],
        default="webster",
        help="Dictionary to use (default: webster)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Starting word for BFS (default: random)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Just download the dictionary data and exit",
    )
    parser.add_argument(
        "--cycle-time",
        type=float,
        default=60.0,
        help="Time budget in seconds for longest-cycle heuristic search (default: 60)",
    )
    parser.add_argument(
        "--fragmentation",
        action="store_true",
        help="Compute island fragmentation at each BFS depth (slow for large dicts)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=WEBSTER_CACHE,
        help=f"Path to cache Webster's JSON (default: {WEBSTER_CACHE})",
    )

    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  Dictionary Graph Explorer")
    print(f"  Dictionary: {args.dict}")
    print(f"{'='*72}\n")

    # ---- Load dictionary ----
    print("─── Loading Dictionary ───")
    t0 = time.time()
    if args.dict == "webster":
        dictionary = load_webster(args.cache_path)
    else:
        dictionary = load_wordnet()
    load_time = time.time() - t0
    print(f"  Loaded {len(dictionary):,} entries in {load_time:.1f}s")
    print()

    if args.download_only:
        print("  --download-only: exiting after download.")
        return

    # ---- Run analysis ----
    run_analysis(
        dict_name=args.dict,
        dictionary=dictionary,
        start_word=args.start,
        cycle_time_limit=args.cycle_time,
        skip_fragmentation=not args.fragmentation,
    )


if __name__ == "__main__":
    main()
