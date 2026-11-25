Questions: 30 · embed: `nomic-embed-text` · k=4 · mode=concise · judge=off

| llm | retrieval | expand | chunk_size | overlap | chunks | hit@1 | hit@4 | MRR | retrieval_ms | context_chars |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | vector | on | 1000 | 150 | 777 | 0.567 | 0.800 | 0.669 | 157.362 | 2463.200 |
| qwen2.5:7b | hybrid | on | 1000 | 150 | 777 | 0.700 | 0.867 | 0.764 | 184.884 | 2520.167 |
