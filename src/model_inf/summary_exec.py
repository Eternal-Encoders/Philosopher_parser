from openai import OpenAI
from tqdm.auto import tqdm


class SummaryExec():
    def __init__(
        self,
        endpoint: str,
        openrouter_key: str,
        name_or_path: str,
        site_url: str | None=None,
        site_title: str | None=None
    ) -> None:
        self.client = OpenAI(
        base_url=endpoint,
        api_key=openrouter_key,
    )
        self.name_or_path = name_or_path
        self.extra_headers = {}

        if site_url is not None:
            self.extra_headers.update({
                'HTTP-Referer': site_url
            })
        if site_title is not None:
            self.extra_headers.update({
                'X-Title': site_title
            })

    def generate_summary_text(self, text: str):
        try:
            response = self.client.chat.completions.create(
                model=self.name_or_path,
                messages=[
                    {"role": "user", "content": f"Сделай краткое саммари следующего текста: {text}"}
                ],
                temperature=0.7,
                max_tokens=200,
            )
            summary_content = response.choices[0].message.content
            return summary_content.strip() if summary_content else text[:100] + "..." if len(text) > 100 else text
        except Exception as e:
            print(f"Ошибка при генерации саммари с помощью LLM: {e}")
            return text[:100] + "..." if len(text) > 100 else text

    def generate_summary(self, texts: list[str]):
        res = []
        for i in tqdm(texts):
            res.append(self.generate_summary_text(i))
        return res