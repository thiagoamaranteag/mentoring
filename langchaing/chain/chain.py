import certifi
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("Um fato curiosos, bem curtinho, sobre {topic}")

class StringFormatterCuston(Runnable):
    def invoke(self, data, config=None):
        print(f"Resposta:\n{data}")

chain = prompt | model | StrOutputParser() | StringFormatterCuston()

chain.invoke({"topic": "carros antigos"})