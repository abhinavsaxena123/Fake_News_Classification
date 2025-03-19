from fastapi import FastAPI
from pydantic import BaseModel
from utils import predict_roberta, predict_distilbert
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title = "Fake NewsClassification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    text: str

@app.post("/predict/roberta")
def classify_roberta(news: NewsRequest):
    prediction = predict_roberta(news.text)
    return {"model": "ROBERTa", "prediction": prediction}


@app.post("/predict/distilbert")
def classify_distilbert(news: NewsRequest):
    prediction = predict_distilbert(news.text)
    return {"model": "DistilBERT", "prediction": prediction}


@app.get("/")
def root():
    return {"message": "Welcome to Fake News Classifier API"}



'''
Steps to Run:
Open Terminal and type:
1)    cd app
2)    .\env\Scripts\activate     (activating venv)
3)    uvicorn main:app --reload
4) Open url 
5) Go to index.html and open it directly in browser.
6) Enter the article to see the prediction.  
'''