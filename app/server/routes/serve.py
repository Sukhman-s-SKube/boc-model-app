from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.server.services.model_service import ModelService, get_model_service

router = APIRouter()


@router.get("/test")
async def test():
    return {"Message": "Serve route is working"}


@router.get("/predict/{date}")
async def predict(date: Optional[str] = None, service: ModelService = Depends(get_model_service)):
    try:
        result = await run_in_threadpool(service.predict, date)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@router.post("/reload")
async def reload_model(service: ModelService = Depends(get_model_service)):
    try:
        payload = await run_in_threadpool(service.reload_latest_model)
        return {"status": "ok", **payload}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to reload model: {exc}",
        ) from exc
