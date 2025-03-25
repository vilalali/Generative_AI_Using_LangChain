from langcain_anthropic import LangChain
from chatmodel import ChatModel
from dotenv import load_dotenv
import os

load_dotenv()

model =ChatAntrhopic(model="gpt-3.5-turbo", temperature=0.5, max_tokens=100)

result=model.invoke("Hello, my name is")

print(result.comment)
