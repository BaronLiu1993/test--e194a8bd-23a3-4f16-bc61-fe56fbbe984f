
from celery import Celery
from openai import OpenAI
from google import genai
from dotenv import load_dotenv 
import base64
from google.genai import types
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String
from prometheus_fastapi_instrumentator import Instrumentator
from typing import Dict, Any
from datetime import datetime
import uuid

import requests
import json

GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

engine = create_engine("postgresql://demo:demo123@localhost:5432/agentinfradb")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy import text
from pgvector.sqlalchemy import Vector 
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Memory(Base):
    __tablename__ = "memory"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input = Column(String, nullable=False)
    prompt = Column(String, nullable=False)
    output = Column(String, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

class Logs(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, nullable=False)
    data = Column(JSONB, nullable=False)
    log_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

def generateEmbeddings(text: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def insertEmbedding(input: str, output: str, prompt: str, node_id: int):
    try:
        with SessionLocal() as session:
            embedding = generateEmbeddings(input)
            
            memoryRow = Memory(
                input=input,
                output=output,
                prompt=prompt,
                embedding=embedding
            )
            
            loggingRow = Logs(
                name="embedding",
                node_id=node_id,
                data={"input": input, "output": output, "prompt": prompt},
                log_type="embedding",
                status="success"
            )
            
            session.add(loggingRow)
            session.add(memoryRow)
            
            session.commit()
            
            session.refresh(loggingRow)
            session.refresh(memoryRow)
            
            return {"success": True, "message": "Successfully Inserted"}
    except Exception as e:
        print(f"Error inserting embedding: {e}")
        return {"success": False, "message": "Internal Server Error"}


def retrieve_memories(thought: str, number_of_memories: int):
    """
    Retrieve the vector embedding for a stored memory.

    This function fetches the vector embedding associated and data on
    it such as the input, output and prompt of the LLM
    """
    query_vector = generateEmbeddings(thought)
    sql = text("""
        SELECT *, embedding <=> :query_vector AS similarity
        FROM memory
        ORDER BY similarity
        LIMIT :number_of_memories
    """)
    with SessionLocal() as session:
        result = session.execute(sql, {
            "query_vector": query_vector,
            "number_of_memories": number_of_memories
        })
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]

brokerLayer = "amqp://guest:guest@localhost:5672//"
cacheLayer = "redis://localhost:6379/0"

celery = Celery("LLMQueue", broker=brokerLayer, backend=cacheLayer)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

def insertLoggingData(name, nodeId, data: Dict[str, Any], logType: str):
    try:
        with SessionLocal() as session:
            loggingRow = Logs(name=name, node_id=nodeId, data=data, log_type=logType, status="success")
            session.add(loggingRow)
            session.commit()
            session.refresh(loggingRow)
            return { "success": True, "message": "Successully Inserted"}
    except Exception as e:
        with SessionLocal() as session:
            loggingRow = Logs(name=name, node_id=nodeId, data=data, log_type=logType, status="failed")
            session.add(loggingRow)
            session.commit()
            session.refresh(loggingRow)   
        return { "success": False, "message": "Internal Server Error" }

code_execution_tool = types.Tool(
    code_execution=types.ToolCodeExecution()
)

google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)


def get_covid_data():
    Data retrieval for pubic Covid DATA

    id = str(uuid.uuid4())
    timestamp = datetime.now()
    try:
        response = requests.get(
            "https://disease.sh/v3/covid-19/all",
            headers={"Authorization": "Bearer "}
        )
        response.raise_for_status()
        data = {
            "id": id,
            "workflow_id": "wf_352337",
            "endpoint": "https://disease.sh/v3/covid-19/all",
            "executed": timestamp,
            "environment": "production",
            "status_code": response.status_code,
            "headers": response.headers,
        }
        insertLoggingData("API", "get_covid_data", data, "GET METHOD")
        return response.json()
    except Exception as e:
        data = {
                "id": id,
                "workflow_id": "wf_352337",
                "endpoint": "https://disease.sh/v3/covid-19/all",
                "executed": timestamp,
                "environment": "production",
                "status_code": response.status_code,
                "headers": response.headers,
            }
        insertLoggingData("API", "get_covid_data", data, "GET METHOD")
        raise Exception(e)

client = genai.Client(api_key=GEMINI_API_KEY)

config = types.GenerateContentConfig(
    tools=[get_covid_data, retrieve_memories],
    system_instruction="You are a public health researcher."
)

contents = [
    types.Content(
        role="user", parts=[types.Part(text="Analyze public health data and return a report of it.")]
    )
]

def scraper(modelInput = ""):
    timestamp = datetime.now()
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config
        )
        
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                tool_call = getattr(part, "function_call", None)
                if not tool_call or not hasattr(tool_call, "name"):
                    continue 

                if part.function_call.name == get_covid_data:
                    result = get_covid_data(**tool_call.args)
                    function_response_part = types.Part.from_function_response(
                        name=part.function_call.name,
                        response={ "result": result }
                    )
                    contents.append(candidate)
                    contents.append(types.Content(role="user", parts=[function_response_part]))

                if part.function_call.name == retrieve_memories:
                    result = retrieve_memories(**tool_call.args)
                    function_response_part = types.Part.from_function_response(
                        name=part.function_call.name,
                        response={ "result": result }
                    )
                    contents.append(candidate)
                    contents.append(types.Content(role="user", parts=[function_response_part]))

        data = {
                "workflow_id": "wf_352337",
                "model": "gemini-2.0-flash",
                "tokens_in": response.usage_metadata.prompt_token_count,
                "token_out": response.usage_metadata.candidates_token_count,
                "token_total": response.usage_metadata.total_token_count,
                "executed": timestamp,
                "environment": "production",
                "status_code": 200,
                "prompt": "You are a public health researcher.Analyze public health data and return a report of it.",
                "input": modelInput,
                "output": response.text,
                "response_size": len(response.text.encode("utf-8"))
            }
        insertLoggingData("LLM", "scraper", data, "LLM Method")
        insertEmbedding(modelInput, response.text, "You are a public health researcher.Analyze public health data and return a report of it.") 
        finalResponse = client.models.generate_content(
            model="gemini-2.0-flash",
            config=config,
            contents=contents
        )
        return finalResponse.text
    except Exception as e:
        raise Exception(e)
