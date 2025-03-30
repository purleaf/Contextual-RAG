from ollama import Client
from pydantic import BaseModel


class ChatOllama:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = Client(host=host)

    def generate_response(self,messages: list[dict], format: dict, model: str):
        try:
            response = self.client.chat(
                messages=messages,
                model=model,
                format=format,
            )
            return response
        except ConnectionError as e:

            return f"ConnectionError: {e}"
        except ValueError as e:

            return f"ValueError: {e}"
        except Exception as e:

            return f"Exception: {e}"
    def generate_message(self, format: dict, model: str, request: str):
        try:
            response = self.client.chat(
                messages=[
                    {
                        'role': 'user',
                        'content': request,
                    }],
                model=model,
                format=format,
            )
            return response
        except ConnectionError as e:

            return f"ConnectionError: {e}"
        except ValueError as e:

            return f"ValueError: {e}"
        except Exception as e:

            return f"Exception: {e}"
