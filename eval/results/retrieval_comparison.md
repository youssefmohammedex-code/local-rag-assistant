Questions: 12 · embed: `nomic-embed-text` · k=4 · mode=concise · judge=off

| llm | retrieval | chunk_size | overlap | chunks | hit@1 | hit@4 | MRR | retrieval_ms |
|---|---|---|---|---|---|---|---|---|
| qwen2.5:7b | vector | 1000 | 150 | 15 | 0.833 | 1.000 | 0.917 | 218.954 |
| qwen2.5:7b | hybrid | 1000 | 150 | 15 | 0.833 | 1.000 | 0.917 | 217.564 |
