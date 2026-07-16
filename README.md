# AnalisisIA

API en **FastAPI** para clasificación (predict) de imágenes.

## Dependencias
Se especifican en `requeriments.txt` (nota: el archivo está escrito como *requeriments*):

- torch
- torchvision
- fastapi
- uvicorn
- python-multipart
- pillow
- numpy
- scikit-learn
- matplotlib
- seaborn

## Instalar
Desde la carpeta `AnalisisIA/`:

```bash
pip install -r requeriments.txt
```

> Si usas `uv`/`uv pip`, puedes instalar igual apuntando a `requeriments.txt`.

## Ejecutar (Uvicorn)
Desde la carpeta `AnalisisIA/`:

```bash
uvicorn app:app --reload
```

## Rutas
- `GET /` → estado de la API
- `POST /predict` → recibe un archivo de imagen (`multipart/form-data`) y responde con el resultado de `predecir()`.
- `http://localhost:8000/docs` → Swagger

