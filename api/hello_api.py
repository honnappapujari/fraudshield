from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message": "hello, fraud API is alive"}
