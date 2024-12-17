from fastapi import APIRouter
from app.services.scraper import scraper

router = APIRouter()

@router.get("/scraper")
def get_data():
    """
    Endpoint para realizar o scraper dos dados.
    """
    return scraper()

# @router.post("/predict")
# def get_prediction(features: dict):
#     """
#     Endpoint para realizar previsões com o modelo ML treinado.
#     """
#     prediction = predict(features)
#     return {"prediction": prediction}
