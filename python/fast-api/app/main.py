from fastapi import FastAPI
from app.api.routers import programmers


app = FastAPI()
app.include_router(
    programmers.router,
    prefix="/api/programmers",
    tags=["programmers"],
)
