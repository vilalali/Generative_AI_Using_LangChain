from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo")

result= llm.invoke("Hello, my name is")

print(result)


