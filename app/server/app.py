from fastapi import FastAPI
from dotenv import load_dotenv
from os import getenv
from fastapi.middleware.cors import CORSMiddleware

from server.routes.serve import router as ServeRouter
from server.utils.s3_util import get_s3

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(ServeRouter, tags=["Serve"],prefix="/api/serve")

@app.get("/", tags=["Root"])
async def root():
    return {"Message": "Server is working"}