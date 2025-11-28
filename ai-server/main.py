from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "AI News Service is running."}