from langchain_google import Google
from dotenv import load_dotenv
import os

load_dotenv()

model = Google(model="gemini", temperature=0.5, max_tokens=100)

result=model.invoke("Hello, my name is")

print(result.comment)


