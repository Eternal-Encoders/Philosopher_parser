# Philosopher Parser

Сервис специализирующийся на RAG для парсинга, структурирования и поиска по документу.

Данный сервис преобразует документ из формата DOCX в иерархический граф, выделяет векторные представления слов и производит поиск по документу.

Сервис поднимает api для использования из кода и моделями в качестве tool'а: REST API и MCP.

## Features

- **Парсинг документов**: Преобразование DOCX файла в MarkDown.
- **OCR**: Распознование из документа изображений и преобразование в текст с помощью OCR.
- **Графовая структура**: Создание NetWorkX графа из документа, сохраняя порядок (Глава -> Подглава -> Информация) и информацию о форматировании.
- **Векторный поиск**: Поиск по смыслу из текста.
- **Генерация Саммари**: Автоматическое сокращение больших чанков документа с помощью LLM.
- **REST API**: FastAPI-based эндпоинты для RAG.
- **MCP Интеграция**: Model Context Protocol server оборачивающий API для моделей.
- **Интерактивная визуализация**: Streamlit приложение для локального использования, для предпросмотра полученного графа.
- **Хранение**: Сохранение необходимой информации, для переиспользование созданного графа.

## Dependencies

- Использует VLM и LLM модели на OpenRouter через OpenAi Client.
- Зависит от модели векторизации Google EmmbeddingGemma 300m, поднятой с помощью VLLM на HuggingFace Spaces.
- Для развертывание компилирует Python код в C с помощью Niutka.
- **Пакетный менеджер**: uv
- **Linter/Formter**: ruff с правилами flake8

## CI

- GitHub Action запуск тестов **PyTest**.
- GitHub Action проверки линтером **Ruff**.
- GitHub Action создания образа для развертывания **Docker** **Nuitka**.

## Project Structure

```
philosopher-parser/
├── src/
│   ├── graphs/                # Ключевой модуль парсинга и поиска
│   │   ├── file_reader.py     # Чтение документа и OCR
│   │   ├── graph_parser.py    # Создание графовой структуры
│   │   ├── retriver.py        # Векторизации и поиск
│   │   └── models.py          # Модели данных
│   ├── model_inf/             # Модуль использования моделей
│   │   ├── ocr_exec.py        # Использование VLM для OCR
│   │   ├── summary_exec.py    # Суммаризация текстов
│   │   └── vectorizer_exec.py # Векторизации текстов
│   ├── mcp/                   # Модуль MCP
│   │   ├── mcp_server.py      # Обертка в Tool'ы
│   │   └── rag_api_client.py  # Переиспользование существующих REST API эндпоинтов
│   └── models.py              # Модели данных
├── tests/                     # Unit тесты
├── main.py                    # FastAPI приложение
└── streamlit_app.py           # Streamlit приложение
```

## Installation

### Setup

**Разработка:**
```bash
uv sync
```

**Развертывание/Тестирование:**
```bash
uv sync --no-dev
```

### Optional Dependencies

**Визуализация**
```bash
uv sync --group view
```

## Configuration

Необходимо создать .env файл по подобию .env.example

```env
# Ключи OpenRouter
OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Ключ к приватному пространству HuggingFace
HUGGINGFACE_HUB_TOKEN="your_hugginface_hub_api_key_here"

# Модели для использования
SUMMARY_MODEL="mistralai/mistral-7b-instruct"
OCR_MODEL="nvidia/nemotron-nano-12b-v2-vl:free"
VECTORIZER_MODEL="google/embeddinggemma-300m"

# Эндпоинты для моделей
SUMMARY_ENDPOINT="https://openrouter.ai/api/v1"
OCR_ENDPOINT="https://openrouter.ai/api/v1"
VECTORIZER_ENDPOINT="https://roaoch-vectorizer.hf.space/v1"
```

## Usage

### 1. FastAPI Server

Старт REST API:

```bash
# Development mode
uv run fastapi dev main.py

# Production mode
uv run python main.py
```

По умолчанию приложение запускается в Prod на 80 порту

#### API Endpoints

**Статус:**
```bash
GET /
GET /health
```

**RAG:**
```bash
POST /rag
Content-Type: application/json

{
  "query": "Какие существуют направления в философии?",
  "top_k": 2,
  "max_length": 2000
}
```

**Открытые вопросы:**
```bash
GET /questions
```
Список открытых вопросов, найденных в тексте.

**Документ:**
```bash
GET /document
```
Передача всего документа в формате MarkDown.

### 2. MCP Server

MCP сервер переиспользует  эндпоинты FastAPI с дополнительными metadata.
- / -> /health
- /search -> /rag

MCP сервер стартует вместе с FastAPI 

### 3. Visualization

```bash
uv run streamlit run streamlit_app.py
```

Элементы:
- **Метрики**: Количество нод, связей, распределение типов и уровней.
- **Интерактивный граф**: Визуализация графа с фильтрами, с помощью PyVis.
- **Поиск и фильтры**: Фильтрация по типу, уровню или ключевому слову
- **Обзоры ноды**: Детализированный просмотр текста, саммари, изображение (если есть) и связанных точек ноды.
- **Экспорт**: Скачать граф как CSV или JSON.

**Важно**: приложение использует граф, из директории __output__, создаваемую запуском main.py.

## Ключевые компоненты

### FileReader (`src/graphs/file_reader.py`)

Препроцессонинг DOCX:
- Конвертирует DOCX в Markdown с помощью `markitdown`.
- Извлечение встроенных изображений.
- Прогоняет OCR по по изображениям.
- Заменяет изображение на распознанный текст.

### GraphParser (`src/graphs/graph_parser.py`)

Преобразование текста в граф:
- Выделяет типы нод: заголовки, простой текст, таблицы, списки, изображения, ссылки.
- Составляет иерархию всего текста.
- Добавляет дополнительные связи между чанками текста.
- Опционально добавляет саммари чанков.

### Retriver (`src/graphs/retriver.py`)

Векторизация и поиск:
- Генерирует эмбеддинги нод.
- Ищет ближайшие к запросу по косинусной близости с помощью einsum.
- Набирает из графа контекст, в которой найденные ноды находятся.
- Форматирует текст и контекст.

### Model Inference Modules

- **OcrExec**: VLM OCR OpenRouter.
- **SummaryExec**: LLM генерация саммари OpenRouter.
- **VectorizerExec**: Векторизация текста HuggingFace Spaces.

## Сохранение

По результату обработки создаются файлы:

```
__output__/                    # Директория сохранения
├── study_fies.md              # MarkDown документ
├── binaries/                  # Директория бинарных файлов
│   ├── graph.pkl              # Граф в формате Pickle
│   ├── docs.pkl               # Numpy массив эмбедингов
│   ├── ids.pkl                # Маппинг индексов эмбеддингов в индексы графа
│   └── images.pkl             # Обработанные изображения
└── media/                     # Директория изображений
    └── image_*.png            # Изображения в формате png
```

## Deploy

Создание Docker образа с Standalone скомпилированным RestAPI и MCP:

```bash
docker build -t philosopher-parser .
docker run -p 80:80 --env-file .env philosopher-parser
```

Использует:
- Multi-stage билд.
- Компиляцию Nuitka.

## Development

### Запуск тестов

```bash
uv run pytest
```
**Важно**: некоторые из тестов, тестируют взаимодействие с OpenRouter и HuggingFace Spaces

### Линтер/Форматирование

Для ruff уже присутствует конфигурации в pyproject.toml.

```bash
uv run ruff check .
uv run ruff format .
```

## TLDR

1. **Document Ingestion**: DOCX file → Markdown (via `markitdown`)
2. **Image Processing**: Extract images → OCR (vision model) → Text embeddings
3. **Graph Construction**: Markdown → Parse nodes → Build NetworkX graph
4. **Embedding Generation**: Node texts → Vector embeddings
5. **Retrieval**: Query → Embedding → Similarity search → Context assembly
