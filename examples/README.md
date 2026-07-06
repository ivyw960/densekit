# Examples

Small, self-contained scripts that run against the NumPy core (no PyTorch, no
downloads).

| Script | What it shows |
| --- | --- |
| `quickstart.py` | Encode a corpus, build a `FlatIndex`, run cosine search |
| `evaluate_beir_like.py` | Encode + index + search a tiny benchmark and score it with recall/nDCG/MRR/MAP |

Run any example from the repository root:

```bash
python examples/quickstart.py
```
