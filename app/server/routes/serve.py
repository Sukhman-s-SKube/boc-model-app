from fastapi import APIRouter, Depends, HTTPException, status, Response

router = APIRouter()

#@route GET api/serve/test
#@description Test serve route
#@access Public
@router.get("/test")
async def test():
    return {"Message": "Serve route is working"}