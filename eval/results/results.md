Questions: 12 · embed: `nomic-embed-text` · llm: `llama3.2:3b` · k=4 · judge=on

| chunk_size | overlap | chunks | hit@1 | hit@4 | MRR | retrieval_ms | faithfulness | correctness |
|---|---|---|---|---|---|---|---|---|
| 300 | 45 | 30 | 1.000 | 1.000 | 1.000 | 1911.926 | 0.750 | 0.917 |
| 600 | 90 | 14 | 1.000 | 1.000 | 1.000 | 1901.838 | 0.917 | 0.917 |
| 1000 | 150 | 8 | 1.000 | 1.000 | 1.000 | 1900.825 | 1.000 | 1.000 |
