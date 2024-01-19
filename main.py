from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import logging
from MoodModel import SpotifyMood

logger = logging.getLogger(__name__)

class Item(BaseModel):
    playlist_url: str
    stress_level: float

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SpotifyMood()

@app.get("/")
async def read_root():
    return JSONResponse(content=jsonable_encoder({"Message": "Yeah you are in the root"}), status_code=200)

@app.post("/predict-mood/")
async def create_item(item: Item):
    data = model.predict(item.playlist_url, item.stress_level)
    return JSONResponse(content=jsonable_encoder(data), status_code=200)
