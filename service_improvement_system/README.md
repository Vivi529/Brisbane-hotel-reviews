# Hotel Service Improvement Strategy System

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place the historical review file and optimization-result file in `./data`.
The historical review file must contain at least:

- `sentence`
- `Sub_Issue`
- `sentiment`

Recommended optional fields are `aspect`, `opinion`, and `ES`.

The evidence-selection implementation uses:

1. score-anchored candidate pools;
2. sentence-level textual informativeness based on effective length and IDF;
3. `sentence-transformers/all-MiniLM-L6-v2` embeddings for semantic representativeness;
4. semantic-neighborhood consistency;
5. role-specific score direction;
6. MMR-based diversity selection.

TF-IDF is retained only for the IDF-based informativeness term and as a fallback representation when Sentence Transformers is disabled.
