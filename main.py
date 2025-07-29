from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.api.routes import router

app = FastAPI()
app.include_router(router)
