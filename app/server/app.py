from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.server.routes.serve import router as ServeRouter
from app.server.services.model_service import get_model_service

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ServeRouter, tags=["Serve"],prefix="/api/serve")


@app.on_event("startup")
async def load_model_on_startup():
    service = get_model_service()
    await run_in_threadpool(service.ensure_model_loaded)


@app.get("/", tags=["Root"])
async def root():
    return {"Message": "Server is working"}
