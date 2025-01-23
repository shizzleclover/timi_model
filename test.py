from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import openai
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from transformers import pipeline

app = FastAPI()
load_dotenv()

mongo_url = os.getenv("MONGO_URL")
open_ai_key = os.getenv("API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"

client = MongoClient(mongo_url)
db = client["case_bud_dev"]
logs_collection = db["logs"]

openai.api_key = open_ai_key

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_api_key

llama_pipe = pipeline("text-generation", model=llama_model_name, use_auth_token=hugging_face_api_key, device=0)

class QueryInput(BaseModel):
    query: str
    user_id: Optional[str] = None
    model_choice: int  

def log_interaction(query: str, response: str, metadata: Dict):
    log_entry = {
        "query": query,
        "response": response,
        "metadata": metadata,
    }
    logs_collection.insert_one(log_entry)

@app.post("/legal-assistant/")
async def legal_assistant(query_input: QueryInput):
    try:
        user_query = query_input.query
        user_id = query_input.user_id
        model_choice = query_input.model_choice

        if model_choice == 1:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal AI assistant. Your role is to assist with legal queries by providing accurate, concise, and context-aware responses."},
                    {"role": "user", "content": user_query},
                ],
            )
            ai_response = response["choices"][0]["message"]["content"]
        elif model_choice == 2:
            legal_context = """
            You are a highly skilled legal assistant with expertise in various legal fields. You assist lawyers by providing comprehensive and accurate legal research, drafting legal documents, and advising on legal matters.
            """
            input_prompt = legal_context + "\n\nUser: " + user_query + "\n\nAssistant:"
            response = llama_pipe(input_prompt, max_new_tokens=300, num_return_sequences=1)
            ai_response = response[0]["generated_text"]
        else:
            raise HTTPException(status_code=400, detail="Invalid model choice")

        metadata = {"user_id": user_id, "model_choice": model_choice} if user_id else {"model_choice": model_choice}
        log_interaction(user_query, ai_response, metadata)
        return {"query": user_query, "response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "running", "message": "Legal AI Assistant is online!"}
