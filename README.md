# Dictionary Graph Explorer

How circular is the dictionary? Every definition is written in words that are
themselves defined, so a dictionary is really a giant directed graph pointing at
itself. This treats it as one and measures it — connectivity, islands, strongly
connected components, and the definitional loops where a chain of definitions
eventually comes back to where it started.

Works on **Webster's 1913** (bundled) or **WordNet** (downloaded on first use).

Made with Claude Opus.

## Usage

```
python dictionary_graph.py --dict webster
python dictionary_graph.py --dict wordnet --start dog
python dictionary_graph.py --dict webster --download-only
python dictionary_graph.py --dict webster --cycle-time 120 --fragmentation
```

| Option | Meaning |
|---|---|
| `--dict {webster,wordnet}` | Which dictionary to analyze (default: `webster`) |
| `--start WORD` | Starting word for the BFS walks (default: random) |
| `--download-only` | Fetch the dictionary data and exit |
| `--cycle-time SECONDS` | Time budget for the longest-cycle heuristic search (default: 60) |
| `--fragmentation` | Compute island fragmentation at each BFS depth (slow on large dictionaries) |
| `--cache-path PATH` | Where to cache Webster's JSON |

## What it reports

- **Scale** — total words (nodes), total edges, average edges per word, and
  self-referential definitions (words that appear in their own definition).
- **Islands** — connected components, their sizes, their share of the whole, and
  sample words from each, plus a size histogram.
- **Directed BFS** from a starting word: how many new words each depth reaches,
  cumulatively and as a percentage, out to depth 10 — i.e. how much of English
  you can unfold from one word by following definitions.
- **Undirected BFS** from the same word, depths 0–4.
- **Island fragmentation** — how much territory remains unexplored at each
  depth of the directed walk.
- **Strongly connected components** — total, non-trivial (size ≥ 2), the size
  distribution, and the ten largest.
- **Definitional loops** — the shortest cycles, all mutual-definition pairs
  (2-cycles, of which Webster's has 45,446), and a heuristic search for the
  longest loop it can find inside the time budget.

Cycle-finding is exact where that is tractable and heuristic where it is not;
the `--cycle-time` budget governs the heuristic half.

Sample output from real runs is checked in as `results_webster.txt` and
`results_wordnet.txt`.

## Data

`webster_1913.json` (bundled) comes from
[matthewreagan/WebstersEnglishDictionary](https://github.com/matthewreagan/WebstersEnglishDictionary)
— Webster's 1913 is public domain. WordNet is downloaded on demand and carries
its own Princeton license.

## License

MIT - see [LICENSE](LICENSE). Free to use, modify and redistribute; provided
as-is, with no warranty.
