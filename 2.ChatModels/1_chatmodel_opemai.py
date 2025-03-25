from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model = chatmodel = OpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=100)

result=model.invoke("Hello, my name is")

print(result.comment)

