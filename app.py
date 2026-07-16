from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tempfile
import os

from inference.predict import predecir

app = FastAPI(
    title="API de Clasificación de Animales",
    version="1.0.0"
)


@app.get("/")
def inicio():
    return {
        "mensaje": "API funcionando correctamente"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:

        contenido = await file.read()

        temp.write(contenido)

        ruta = temp.name

    try:

        resultado = predecir(ruta)

        return resultado

    finally:

        if os.path.exists(ruta):
            os.remove(ruta)