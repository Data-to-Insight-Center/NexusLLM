import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

# get the env variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PWD")

# load the graph and the LLM
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)
# gpt4
llm = ChatOpenAI(temperature=0.2, model="gpt-4o")

# llama 3.1 8b
llama31 = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

top_k_results = 10