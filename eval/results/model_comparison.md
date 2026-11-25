Questions: 12 · embed: `nomic-embed-text` · k=4 · mode=concise · judge=on (qwen2.5:7b)

| llm | chunk_size | overlap | chunks | hit@1 | hit@4 | MRR | retrieval_ms | faithfulness | correctness | answer_s |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 1000 | 150 | 15 | 1.000 | 1.000 | 1.000 | 1906.769 | 0.917 | 1.000 | 11.649 |
| qwen2.5:7b | 1000 | 150 | 15 | 1.000 | 1.000 | 1.000 | 2069.617 | 1.000 | 1.000 | 29.250 |
