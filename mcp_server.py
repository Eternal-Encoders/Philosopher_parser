import os
from fastmcp import FastMCP
from mcp.types import TextContent
import httpx
import atexit
import asyncio
from dotenv import load_dotenv


load_dotenv()

# Создаем MCP сервер
app = FastMCP(
    name="Philosopher RAG Client",
    # host="localhost",
    port=8001,
)

# Конфигурация
API_BASE_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
API_TIMEOUT = 30

class RAGAPIClient:
    """Клиент для работы с RAG API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=API_TIMEOUT)
    
    async def search(self, query: str) -> dict:
        """
        Выполняет RAG поиск через API
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Ответ от API в формате словаря
        """
        top_k = 3
        max_length = 4000
        try:
            url = f"{self.base_url}/rag"
            payload = {
                "query": query,
                "top_k": top_k,
                "max_length": max_length
            }
            
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.ConnectError:
            return {
                "error": f"Не удалось подключиться к RAG API по адресу {self.base_url}",
                "docs": ["Проверьте, запущен ли FastAPI сервер на порту 8000"],
                "meta": {"status": "connection_error"}
            }
        except httpx.HTTPStatusError as e:
            return {
                "error": f"Ошибка API: {e.response.status_code}",
                "docs": [f"Подробности: {e.response.text}"],
                "meta": {"status": "http_error"}
            }
        except Exception as e:
            return {
                "error": f"Неизвестная ошибка: {str(e)}",
                "docs": [],
                "meta": {"status": "error"}
            }
    
    async def health_check(self) -> bool:
        """Проверяет доступность API"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()

# Инициализируем клиент
rag_client = RAGAPIClient()

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

@app.tool(name='search')
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

@app.tool()
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


if __name__ == "__main__":
    # Запускаем MCP сервер
    app.run(transport="http")
