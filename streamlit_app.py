"""Streamlit application to visualize parsed philosophical text graphs and their summaries.

Run with:
    streamlit run streamlit_app.py

The app loads pickled NetworkX graph and node data produced by the parsing pipeline:
    __output__/binaries/graph.pkl
    __output__/binaries/nodes_data.pkl

Features:
- Overview metrics (nodes, edges, type distribution, level stats).
- Interactive graph visualization (PyVis) with filtering.
- Node search by keyword in text/summary.
- Filter by node type and level range.
- Inspect node details (text, summary, image if present, neighbors).
- Download node data as JSON / CSV.
"""
from __future__ import annotations
import os
import io
import json
import pickle
import base64
from datetime import datetime
from typing import Any, Dict, List, Tuple

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from PIL import Image

GRAPH_FILE = os.path.join("__output__", "binaries", "graph.pkl")
IDS_FILE = os.path.join("__output__", "binaries", "ids.pkl")

TYPE_COLOR_MAP = {
    "heading": "#1f77b4",
    "text": "#2ca02c",
    "table": "#ff7f0e",
    "list": "#9467bd",
    "image": "#d62728",
    "footnote": "#17becf",
}


def normalize_node_type(t: Any) -> str:
    """Return a stable string type for nodes regardless of Enum/str representation.
    Examples:
      - TypeReturn.HEADING -> "heading"
      - "TypeReturn.HEADING" -> "heading"
      - "heading" -> "heading"
      - None -> "unknown"
    """
    if t is None:
        return "unknown"
    # Enum-like with .value
    val = getattr(t, "value", None)
    if isinstance(val, str):
        return val
    s = str(t)
    if s.startswith("TypeReturn."):
        return s.split(".")[-1].lower()
    return s.lower()

def _file_fingerprint(path: str) -> Tuple[int, float]:
    """Return (size, mtime) for invalidation purposes."""
    if not os.path.exists(path):
        return (0, 0.0)
    return (os.path.getsize(path), os.path.getmtime(path))

@st.cache_data(show_spinner=False)
def load_ids(path: str = IDS_FILE, fingerprint: Tuple[int, float] | None = None) -> List[Any]:
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_graph(path: str = GRAPH_FILE, fingerprint: Tuple[int, float] | None = None) -> nx.Graph | None:
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def nodes_from_graph(graph: nx.Graph, ids: List[Any] | None = None) -> List[Dict[str, Any]]:
    """Construct nodes_data from graph nodes, optionally filtered/ordered by ids.
    Format: [{'id': k, **v} for k, v in dict(graph.nodes.data()).items()].
    """
    gdict: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in dict(graph.nodes.data()).items()}
    if ids is not None and len(ids) > 0:
        result: List[Dict[str, Any]] = []
        for raw_id in ids:
            sid = str(raw_id)
            if sid in gdict:
                entry = {'id': sid}
                entry.update(gdict[sid])
                result.append(entry)
        return result
    return [{'id': k, **v} for k, v in gdict.items()]

def load_artifacts() -> Tuple[nx.Graph | None, List[Dict[str, Any]], Tuple[int,float], Tuple[int,float]]:
    graph_fp = _file_fingerprint(GRAPH_FILE)
    ids_fp = _file_fingerprint(IDS_FILE)
    graph = load_graph(GRAPH_FILE, graph_fp)
    ids = load_ids(IDS_FILE, ids_fp)
    nodes_data: List[Dict[str, Any]] = []
    if graph is not None:
        nodes_data = nodes_from_graph(graph, ids if len(ids) > 0 else None)
    nodes_fp = ids_fp if ids_fp != (0, 0.0) else graph_fp
    return graph, nodes_data, graph_fp, nodes_fp

def compute_overview(nodes: List[Dict[str, Any]], graph: nx.Graph | None) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    levels: List[int] = []
    for n in nodes:
        t = normalize_node_type(n.get('node_type'))
        by_type[t] = by_type.get(t, 0) + 1
        lvl = n.get('level')
        if isinstance(lvl, int):
            levels.append(lvl)
    return {
        'total_nodes': len(nodes),
        'total_edges': graph.number_of_edges() if graph else 0,
        'types': by_type,
        'min_level': min(levels) if levels else None,
        'max_level': max(levels) if levels else None,
        'avg_level': sum(levels)/len(levels) if levels else None,
    }

def node_passes_filters(node: Dict[str, Any], keyword: str, types: List[str], level_range: Tuple[int, int]) -> bool:
    node_type = normalize_node_type(node.get('node_type'))
    if types and node_type not in types:
        return False
    lvl = node.get('level', -1)
    if isinstance(lvl, int) and not (level_range[0] <= lvl <= level_range[1]):
        return False
    if keyword:
        blob = (node.get('text', '') + ' ' + str(node.get('summary', ''))).lower()
        if keyword.lower() not in blob:
            return False
    return True

@st.cache_data(show_spinner=False)
def build_pyvis_html(
    nodes_snapshot: Tuple[Tuple[str,str,int,str,str|None], ...],
    edges_snapshot: Tuple[Tuple[str,str], ...],
    keyword: str,
    selected_types: Tuple[str, ...],
    level_range: Tuple[int, int]
) -> str:
    """Cached HTML graph build based on immutable snapshots and filters."""
    # Reconstruct lightweight dict list for filtering
    nodes_data = [
        {
            'id': nid,
            'node_type': ntype,
            'level': lvl,
            'text': text,
            'summary': summary
        }
        for (nid, ntype, lvl, text, summary) in nodes_snapshot
    ]
    allowed_ids = {
        n['id'] for n in nodes_data
        if node_passes_filters(n, keyword, list(selected_types), level_range)
    }
    net = Network(height="650px", width="100%", bgcolor="#FFFFFF", directed=False, notebook=False)
    net.barnes_hut()
    for n in nodes_data:
        if n['id'] not in allowed_ids:
            continue
        label = n['text'][:60] + ('…' if len(n['text']) > 60 else '')
        title_parts = [
            f"<b>ID:</b> {n['id']}",
            f"<b>Type:</b> {n['node_type']}",
            f"<b>Level:</b> {n['level']}",
        ]
        if n.get('summary'):
            summary = n['summary'] or ''
            title_parts.append(f"<b>Summary:</b> {summary[:200]}{'…' if len(summary) > 200 else ''}")
        color = TYPE_COLOR_MAP.get(n['node_type'], '#888888')
        net.add_node(
            n['id'],
            label=label if label else n['node_type'],
            title="<br/>".join(title_parts),
            color=color,
            shape='dot',
            size=12 + (n['level'] or 0) * 2,
        )
    for a, b in edges_snapshot:
        if a in allowed_ids and b in allowed_ids:
            net.add_edge(a, b)
    return net.generate_html()

def make_graph_snapshots(graph: nx.Graph, nodes_data: List[Dict[str, Any]]):
    nodes_snapshot_list: List[Tuple[str, str, int, str, str | None]] = []
    for n in nodes_data:
        nid = str(n['id'])
        ntype = normalize_node_type(n.get('node_type'))
        lvl_raw = n.get('level')
        lvl = int(lvl_raw) if isinstance(lvl_raw, int) else -1
        text = str(n.get('text', '') or '')
        summ_val = n.get('summary')
        summary: str | None = str(summ_val) if isinstance(summ_val, str) else None
        nodes_snapshot_list.append((nid, ntype, lvl, text, summary))
    nodes_snapshot = tuple(nodes_snapshot_list)
    edges_snapshot = tuple((str(a), str(b)) for a, b in graph.edges())
    return nodes_snapshot, edges_snapshot

def download_nodes(nodes: List[Dict[str, Any]], fmt: str) -> bytes:
    """Return file bytes encoded in UTF-8 for JSON and UTF-8 with BOM for CSV (Excel-friendly)."""
    if fmt == 'json':
        # Pure UTF-8 without BOM (standard for JSON)
        return json.dumps(nodes, ensure_ascii=False, indent=2).encode('utf-8')
    if fmt == 'csv':
        # Add BOM via utf-8-sig so Excel on Windows opens Cyrillic correctly
        import csv
        output = io.StringIO(newline='')
        fieldnames = sorted({k for n in nodes for k in n.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for n in nodes:
            writer.writerow(n)
        return output.getvalue().encode('utf-8-sig')
    raise ValueError('Unsupported format')

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def main():
    st.set_page_config(page_title="Philosopher Graph Explorer", layout="wide", page_icon="🧠")
    st.title("🧠 Philosopher Graph Explorer")
    st.caption("Explore parsed structure, summaries, and relationships.")

    graph, nodes_data, graph_fp, nodes_fp = load_artifacts()

    # Sidebar control: limit number of nodes for quick testing (0 = all)
    max_nodes = st.sidebar.number_input(
        "Max nodes to display (0 = all, for testing)",
        min_value=0,
        max_value=max(0, len(nodes_data)),
        value=0,
        step=1,
    )
    if max_nodes > 0 and len(nodes_data) > max_nodes:
        nodes_data = nodes_data[:max_nodes]

    if graph is None or not nodes_data:
        st.error("Graph or IDs not found. Please generate '__output__/binaries/graph.pkl' and 'ids.pkl'.")
        st.stop()

    overview = compute_overview(nodes_data, graph)
    with st.expander("Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", overview['total_nodes'])
        col2.metric("Edges", overview['total_edges'])
        col3.metric("Levels (min/max)", f"{overview['min_level']} / {overview['max_level']}")
        avg_lvl = overview['avg_level']
        col4.metric("Avg Level", f"{avg_lvl:.2f}" if avg_lvl is not None else "-")
        st.write("**Type distribution**")
        dist_cols = st.columns(len(overview['types'])) if overview['types'] else []
        for (t, cnt), c in zip(overview['types'].items(), dist_cols):
            c.metric(t, cnt)
        
        # Display file modification timestamps
        st.markdown("---")
        st.caption("**Data freshness:**")
        if graph_fp[1] > 0:
            graph_time = datetime.fromtimestamp(graph_fp[1]).strftime('%Y-%m-%d %H:%M:%S')
            st.caption(f"Graph: {graph_time}")
        if nodes_fp[1] > 0:
            nodes_time = datetime.fromtimestamp(nodes_fp[1]).strftime('%Y-%m-%d %H:%M:%S')
            st.caption(f"Nodes: {nodes_time}")

    st.sidebar.header("Filters")
    all_types = sorted({normalize_node_type(n.get('node_type')) for n in nodes_data})
    selected_types = st.sidebar.multiselect("Node types", all_types, default=all_types)

    min_lvl = overview['min_level'] if overview['min_level'] is not None else -1
    max_lvl = overview['max_level'] if overview['max_level'] is not None else 5
    level_range = st.sidebar.slider("Level range", min_lvl, max_lvl, (min_lvl, max_lvl))

    keyword = st.sidebar.text_input("Keyword search (text/summary)")

    # Reset filters button
    if st.sidebar.button("Сбросить фильтры", help="Вернуть все фильтры к начальным значениям"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    # Refresh caches and reload artifacts
    if st.sidebar.button("Обновить граф", help="Очистить кэш и перечитать бинарные файлы графа"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        finally:
            # Перезапускаем приложение, чтобы перечитать данные
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.write("Download node data:")
    fmt = st.sidebar.selectbox("Format", ["json", "csv"])
    if st.sidebar.button("Download"):
        data_bytes = download_nodes(nodes_data, fmt)
        st.sidebar.download_button(
            "Save file",
            data_bytes,
            file_name=f"nodes_data.{fmt}",
            mime=("application/json; charset=utf-8" if fmt == 'json' else 'text/csv; charset=utf-8')
        )

    st.subheader("Interactive Graph")
    # Graph is guaranteed non-None beyond this point due to earlier stop,
    # add assertion for type-checkers.
    assert graph is not None
    nodes_snapshot, edges_snapshot = make_graph_snapshots(graph, nodes_data)
    html_graph = build_pyvis_html(
        nodes_snapshot,
        edges_snapshot,
        keyword,
        tuple(selected_types),
        level_range
    )
    selectable_ids = [
        n['id'] for n in nodes_data
        if node_passes_filters(n, keyword, selected_types, level_range)
    ]
    if len(selectable_ids) == 0:
        st.warning("Нет узлов для отображения с текущими фильтрами.")
    else:
        components.html(html_graph, height=650, scrolling=True)

    st.subheader("Node Inspector")
    selected_node_id = st.selectbox("Select node", options=selectable_ids)

    if selected_node_id:
        node_info = next((n for n in nodes_data if n['id'] == selected_node_id), None)
        if node_info:
            st.markdown(f"**Type:** `{node_info.get('node_type')}` | **Level:** `{node_info.get('level')}` | **Position:** `{node_info.get('position')}`")
            st.markdown(f"**Text:**\n\n{node_info.get('text')}")
            if node_info.get('summary'):
                st.markdown(f"**Summary:**\n\n> {node_info.get('summary')}")
            # Image display if attached via graph (graph nodes store image object or None)
            gnode = graph.nodes.get(selected_node_id)
            if gnode and gnode.get('image') is not None:
                img: Image.Image = gnode['image']
                st.image(img, caption="Node image", use_container_width=True)

            # Neighbor details
            st.markdown("**Neighbors:**")
            neighbor_ids = list(graph.neighbors(selected_node_id))
            if not neighbor_ids:
                st.write("No neighbors.")
            else:
                for nid in neighbor_ids:
                    nd = graph.nodes[nid]
                    summary = nd.get('summary') or ''
                    text_val = nd.get('text') or ''
                    st.markdown(f"- `{nid[:8]}` | *{nd.get('node_type')}* | {text_val[:80]}{'…' if len(text_val)>80 else ''}")
                    if summary:
                        st.caption(summary[:160] + ('…' if len(summary) > 160 else ''))

    st.markdown("---")
    st.caption("Data loaded from pickled artifacts. Re-run parser to refresh.")

if __name__ == "__main__":
    main()
