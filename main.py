from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import os
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
import models
from database import engine, get_db
models.Base.metadata.create_all(bind=engine)

# ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────
# FIX 3: PATH CONFIGURATION ERRORS
# Replaced hardcoded "D:\" paths with a clean, dynamic relative path 
# so it works on Linux cloud hosting servers.
# ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fogmap_efficientnet_v2.keras")

app = FastAPI(title="Fog Detection API")

# Enable CORS middleware so your frontend can communicate with the hosted API URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────
# FIX 1: REMOVED DEPLOYMENT BLOCKERS (matplotlib)
# Completely removed all 'plt.show()' and 'plt.imshow()' lines 
# to prevent the headless cloud server from crashing.
# ────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Warm up the model using the direct execution syntax to optimize first-run speed
dummy_input = np.zeros((1, 224, 224, 3))
model(dummy_input, training=False) 
print("Model ready for hosting!")

@app.post("/predict")
async def predict_fog(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        # Read the uploaded image file safely into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        # ────────────────────────────────────────────────────────
        # FIX 4: THE PIXEL NORMALIZATION TRAP
        # Kept the array raw between 0 and 255. No manual division (/ 255.0) 
        # because the EfficientNet architecture handles rescaling internally.
        # ────────────────────────────────────────────────────────
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # ────────────────────────────────────────────────────────
        # FIX 2: SERVER PERFORMANCE BUG (model.predict)
        # Swapped model.predict() out for direct tensor calling syntax.
        # This keeps multi-threaded web requests smooth and fast.
        # ────────────────────────────────────────────────────────
        prediction_tensor = model(img_array, training=False)
        prediction = prediction_tensor.numpy()[0][0]
        
        # Classification thresholds
        if prediction > 0.50:
            label = "Smog"
            confidence = float(prediction)
        else:
            label = "Clear"
            confidence = float(1 - prediction)
            
        process_time_ms = round((time.time() - start_time) * 1000, 2)
        confidence_pct = round(confidence * 100, 2)

        # ---> ADD THESE DB TRANSACTION LINES <---
        if label == "Smog":
            visibility = "Low" if confidence_pct >= 75.0 else "Moderate"
            is_alert_needed = True if confidence_pct >= 65.0 else False
        else:
            visibility = "High"
            is_alert_needed = False

        db_detection = models.FogDetection(
            prediction_label=label,
            confidence_score=confidence_pct,
            visibility_level=visibility,
            alert_sent=is_alert_needed,
            latency_ms=process_time_ms,
            camera_location="Van_Camera_01"
        )
        db.add(db_detection)
        db.commit()
        db.refresh(db_detection)

        alert_triggered = None
        if is_alert_needed:
            level = "Critical" if confidence_pct >= 85.0 else "Warning"
            db_alert = models.ActiveAlert(
                detection_id=db_detection.id,
                alert_level=level,
                is_resolved=False
            )
            db.add(db_alert)
            db.commit()
            db.refresh(db_alert)
            alert_triggered = {"alert_id": db_alert.alert_id, "level": db_alert.alert_level}

        # Updated JSON Output mapping your notebook fields
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": label,
            "confidence_percent": confidence_pct,
            "visibility": visibility,
            "alert_sent": is_alert_needed,
            "latency_ms": process_time_ms,
            "db_record_id": db_detection.id,
            "alert_info": alert_triggered
        }
    except Exception as e:
        db.rollback() # ---> ADD THIS LINE HERE
        return {"status": "error", "message": str(e)}
    
    # ---> PASTE THESE THREE ENDPOINTS HERE <---

@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.FogDetection).order_by(models.FogDetection.id.desc()).limit(limit).all()

@app.put("/alerts/{alert_id}")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(models.ActiveAlert).filter(models.ActiveAlert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert record not found")
    alert.is_resolved = True
    db.commit()
    return {"status": "success", "message": f"Alert {alert_id} resolved."}

@app.delete("/logs/old")
def purge_old_records(db: Session = Depends(get_db)):
    db.query(models.ActiveAlert).delete()
    deleted_count = db.query(models.FogDetection).delete()
    db.commit()
    return {"status": "success", "records_purged": deleted_count}

# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

# Dynamic port assignment hook so cloud providers (like Render) can host it perfectly
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
