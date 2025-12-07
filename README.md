# Philosopher_parser

**For running experiments with local python models**
```bash
uv sync
```

**For remote models and general usage**
```bash
uv sync --no-dev
uv run fastapi dev
```

**Run the Streamlit UI**

The repository includes a Streamlit-based explorer for the parsed graph. Start it with:

```bash
uv run streamlit run streamlit_app.py
```

Or run Streamlit directly (if `uv` runner is not used):

```bash
streamlit run streamlit_app.py
```