import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

checkpoint = torch.load(
    "models/modelo_final.pth",
    map_location=DEVICE
)

clases = checkpoint["clases"]

modelo = resnet18(weights=None)

modelo.fc = nn.Linear(
    modelo.fc.in_features,
    len(clases)
)

modelo.load_state_dict(
    checkpoint["modelo"]
)

modelo.to(DEVICE)

modelo.eval()

def predecir(ruta_imagen):

    imagen = Image.open(ruta_imagen).convert("RGB")

    imagen = transform(imagen)

    imagen = imagen.unsqueeze(0)

    imagen = imagen.to(DEVICE)

    with torch.no_grad():

        salida = modelo(imagen)

        probabilidades = torch.softmax(salida, dim=1)

        confianza, indice = torch.max(probabilidades, 1)

    return {
        "animal": clases[indice.item()],
        "confianza": round(confianza.item() * 100, 2)
    }


