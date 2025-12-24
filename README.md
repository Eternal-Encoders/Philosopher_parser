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

**Run the MCP server (RAG tools)**

Start the Model Context Protocol server that exposes RAG tools (`search`). It runs over stdio using FastMCP.

Start MCP server for dev
```bash
uv run fastmcp dev src/mcp/mcp_server.py
```

**Run the Streamlit UI**

The repository includes a Streamlit-based explorer for the parsed graph. Start it with:

```bash
uv run streamlit run streamlit_app.py
```
