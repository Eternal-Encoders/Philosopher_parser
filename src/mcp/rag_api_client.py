import httpx


class RAGAPIClient:
    """Клиент для работы с RAG API"""
    
    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=timeout)
    
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
                "error": f"Не удалось подключиться к RAG API \
                    по адресу {self.base_url}",
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
        except Exception as e:
            print(e)
            return False
    
    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()