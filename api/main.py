from fastapi import FastAPI, UploadFile, File ,Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from database import PredictionRecord, get_db


# ==== CONFIGURATION ====
IMG_SIZE = 256
CLASS_NAMES = ['Early_Blight', 'Late_Blight', 'Healthy']
MODEL_PATH = "models/1/v1.keras"


# ==== LOAD MODEL ====
model = tf.keras.models.load_model(MODEL_PATH)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== AUTO-DETECT IF MODEL HAS Rescaling LAYER ====
def model_has_rescaling_layer(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
        elif hasattr(layer, "layers"):  # if it's a Sequential block
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Rescaling):
                    return True
    return False

HAS_RESCALING = model_has_rescaling_layer(model)

# ==== IMAGE PREPROCESSING ====
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image).astype(np.float32)

    if not HAS_RESCALING:
        image_array = image_array / 255.0  # normalize only if model doesn't

    image_tensor = np.expand_dims(image_array, axis=0)  # shape: (1, 256, 256, 3)
    return image_tensor

# ==== PREDICTION ROUTE ====
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    contents = await file.read()
    image_tensor = preprocess_image(contents)

    predictions = model.predict(image_tensor)
    confidence = float(np.max(predictions[0]))*100
    predicted_class = CLASS_NAMES[int(np.argmax(predictions[0]))]
    # Save prediction to database
    record = PredictionRecord(
        filename=file.filename,
        predicted_class=predicted_class,
        confidence=round(confidence, 2)
    )
    db.add(record)
    await db.commit()
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        
    }
