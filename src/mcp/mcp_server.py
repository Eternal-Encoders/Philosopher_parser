import os
import atexit
import asyncio
from fastmcp import FastMCP
from mcp.types import TextContent
from dotenv import load_dotenv
from .rag_api_client import RAGAPIClient

load_dotenv()

# Создаем MCP сервер
mcp_app = FastMCP(
    name="Philosopher RAG Client"
)

# Конфигурация
API_BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Инициализируем клиент
rag_client = RAGAPIClient(API_BASE_URL, API_TIMEOUT)

# Функция для закрытия клиента при завершении
def cleanup():
    """Очистка ресурсов при завершении"""
    print("\n🧹 Cleaning up resources...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_client.close())
        loop.close()
    except:
        pass

# Регистрируем функцию очистки
atexit.register(cleanup)

@mcp_app.tool(name='search')
async def search(query: str) -> list[TextContent]:
    """
    Поиск философских текстов по запросу.
    
    Этот инструмент использует внешний RAG API для поиска релевантных документов.
    
    Args:
        query: Ваш вопрос или тема для поиска
    
    Returns:
        Найденные документы с метаинформацией
    """
    
    try:
        # Вызываем API
        result = await rag_client.search(query=query)
        
        # Проверяем наличие ошибок
        if "error" in result:
            error_text = f"⚠️ Ошибка: {result['error']}\n\n"
            if result.get("docs"):
                error_text += "Документы:\n" + "\n\n".join(result["docs"])
            return [TextContent(type="text", text=error_text)]
        
        # Форматируем ответ
        docs = result.get("docs", [])
        meta = result.get("meta", {})
        
        if not docs:
            return [TextContent(type="text", text=f"📭 По запросу '{query}' ничего не найдено")]
        
        # Создаем форматированный ответ
        response_parts = []
        
        # Заголовок
        response_parts.append(f"🔍 **Результаты поиска:** '{query}'")
        response_parts.append("---")
        
        # Документы
        for i, doc in enumerate(docs, 1):
            response_parts.append(f"**Документ {i}:**")
            response_parts.append(doc)
            response_parts.append("")
        
        # Метаинформация
        if meta:
            response_parts.append("📊 **Метаинформация:**")
            for key, value in meta.items():
                response_parts.append(f"  • {key}: {value}")
        
        response_text = "\n".join(response_parts)
        return [TextContent(type="text", text=response_text)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Ошибка при поиске: {str(e)}")]

@mcp_app.tool()
async def check_api_status() -> list[TextContent]:
    """
    Проверить статус RAG API сервера.
    
    Returns:
        Статус подключения к API серверу
    """
    try:
        is_healthy = await rag_client.health_check()
        
        if is_healthy:
            return [TextContent(
                type="text", 
                text=f"✅ RAG API сервер доступен по адресу: {API_BASE_URL}\n\n"
                     f"Для поиска используйте инструмент 'search_philosophy'"
            )]
        else:
            return [TextContent(
                type="text",
                text=f"⚠️ RAG API сервер недоступен по адресу: {API_BASE_URL}\n\n"
                     f"Убедитесь, что FastAPI сервер запущен."
            )]
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Ошибка проверки статуса: {str(e)}")]
