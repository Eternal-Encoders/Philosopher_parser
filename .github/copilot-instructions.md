# Copilot Instructions for Philosopher_parser

This document provides essential guidelines for AI coding agents working on the `Philosopher_parser` codebase.

## 1. Project Overview
`Philosopher_parser` is a Python application designed to parse and process data, likely from philosophical texts, and represent them in a graph structure.

## 2. Core Architecture and Data Flow

### Main Components:
*   `main.py`: The primary entry point for the application.
*   `src/connector.py`: Manages connections to external data sources or APIs.
*   `src/models.py`: Defines the core data models used across the application.
*   `src/graphs/`: This sub-package is responsible for all graph-related operations:
    *   `file_reader.py`: Handles reading and initial processing of input files.
    *   `graph_parser.py`: Parses the processed data into a graph structure.
    *   `models.py`: Contains data models specific to the graph representation.
    *   `retriver.py`: Likely retrieves additional data needed for graph construction or enrichment.
    *   `utils.py`: Provides utility functions for the `graphs` component.

### Data Flow:
1.  Input data (e.g., text files) is read via `src/graphs/file_reader.py`.
2.  Data might be further processed or enriched using `src/connector.py` or `src/graphs/retriver.py`.
3.  The processed data is then transformed into a graph structure by `src/graphs/graph_parser.py`, utilizing models defined in `src/graphs/models.py`.

## 3. Critical Developer Workflows

### Dependency Management:
The project uses `uv` for Python package management.
*   To synchronize dependencies, run: `uv sync`

## 4. Project-Specific Conventions
*   **Modularity:** The codebase is structured into logical modules (`src/`, `src/graphs/`) to promote separation of concerns.
*   **Graph-centric Processing:** The `src/graphs/` directory is central to the project's data processing paradigm, focusing on graph creation and manipulation.

## 5. Integration Points
*   External data sources are likely integrated through `src/connector.py`.
*   The graph processing pipeline in `src/graphs/` is a key integration point for various data inputs and potential downstream graph analysis tools.

Please provide feedback if any sections are unclear or incomplete.