from typing import Literal, Optional
import os
from dotenv import load_dotenv
from openai import chat
from utils.config_loader import load_config
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

class ConfigLoader:
    def __init__(self):
        print("Loading config...")
        self.config = load_config()
        
    def __getitem__(self, key):
        return self.config[key]

class ModelLoader(BaseModel):
    model_provider:Literal["groq","openai"] = "groq"
    config: Optional[ConfigLoader] = Field(default=None,exclude=True)
    
    def model_post_init(self):
        self.config = ConfigLoader()
        
    class config:
        arbitrary_types_allowed = True
        
    def load_llm(self):
        print(f"Loading {self.model_provider} model...")
        
        if self.model_provider=="groq":
            groq_api_key= os.getenv("GROQ_API_KEY")
            model_name= self.config["llm"]["groq"]["model_name"]
            llm=ChatGroq(model=model_name,api_key=groq_api_key)
        
        elif self.model_provider=="openai":
            openai_api_key= os.getenv("OPENAI_API_KEY")
            model_name= self.config["llm"]["openai"]["model_name"]
            llm=ChatOpenAI(model=model_name,api_key=openai_api_key)
        
        return llm
            
            
    