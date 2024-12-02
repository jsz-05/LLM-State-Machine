import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

client = openai.OpenAI()
models_page = client.models.list()

for model in models_page:
    print(model.id)
