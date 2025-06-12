from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca_components.pkl', 'rb') as f:
        pca_components = pickle.load(f)
    with open('pca_mean.pkl', 'rb') as f:
        pca_mean = pickle.load(f)
    with open('lda.pkl', 'rb') as f:
        lda = pickle.load(f)
    logger.info("Models loaded successfully")
    logger.info(f"PCA components shape: {pca_components.shape}, LDA classes: {lda.classes_}")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading models: {e}")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    logger.info("Received POST request to /predict")
    try:
        contents = await image.read()
        logger.info(f"Received image of size {len(contents)} bytes")
        if not contents:
            logger.error("Empty image received")
            raise HTTPException(status_code=422, detail="Empty image received")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=422, detail="Cannot decode image")
        
        logger.info(f"Decoded image shape: {img.shape}")
        
        img_resized = cv2.resize(img, (100, 56))
        img_flat = img_resized.flatten().reshape(1, -1)
        logger.info(f"Image resized to {img_resized.shape}, flattened to {img_flat.shape}")
        
        # نرمال‌سازی اولیه
        img_flat = img_flat / 255.0
        
        # استانداردسازی با MinMaxScaler
        img_scaled = scaler.transform(img_flat)
        
        expected_features = 100 * 56 * 3
        if img_flat.shape[1] != expected_features:
            logger.error(f"Unexpected image shape: {img_flat.shape}")
            raise HTTPException(status_code=422, detail=f"Expected {expected_features} features, got {img_flat.shape[1]}")
        
        if pca_components.shape[0] != img_flat.shape[1]:
            logger.error(f"PCA components shape {pca_components.shape[0]} does not match image features {img_flat.shape[1]}")
            raise HTTPException(status_code=422, detail="PCA components shape mismatch")
        
        img_pca = np.dot(img_scaled - pca_mean, pca_components)
        pred_idx = lda.predict(img_pca)[0]
        
        # نگاشت به برچسب‌های واقعی
        #label_map = {0: '5₺', 1: '10₺', 2: '20₺', 3: '50₺', 4: '100₺', 5: '200₺'}
        label_map = {5: '5₺', 10: '10₺', 20: '20₺', 50: '50₺', 100: '100₺', 200: '200₺'}
        if pred_idx not in label_map:
            logger.error(f"Invalid prediction index: {pred_idx}, expected one of {list(label_map.keys())}")
            raise HTTPException(status_code=422, detail=f"Invalid prediction: {pred_idx}")
        
        prediction = label_map[pred_idx]
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error processing image: {str(e)}")  

@app.post("/predict2")
async def predict_image(image: UploadFile = File(...)):
    """
    Controller to receive a JPG image, process it, and predict the Turkish banknote class using saved models.
    Returns the predicted banknote class (e.g., '5₺').
    """
    # تنظیم لاگ
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Received POST request to /predict_image")

    # لود مدل‌های ذخیره‌شده
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('pca_components.pkl', 'rb') as f:
            pca_components = pickle.load(f)
        with open('pca_mean.pkl', 'rb') as f:
            pca_mean = pickle.load(f)
        with open('lda.pkl', 'rb') as f:
            lda = pickle.load(f)
        logger.info("Models loaded successfully")
        logger.info(f"PCA components shape: {pca_components.shape}, LDA classes: {lda.classes_}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading models: {e}")

    # خواندن و پردازش تصویر
    try:
        # خواندن تصویر
        contents = await image.read()
        logger.info(f"Received image of size {len(contents)} bytes")
        if not contents:
            logger.error("Empty image received")
            raise HTTPException(status_code=422, detail="Empty image received")

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=422, detail="Cannot decode image")

        logger.info(f"Decoded image shape: {img.shape}")

        # تبدیل BGR به RGB (چون OpenCV به‌صورت BGR می‌خونه)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # تغییر اندازه به 100x56 (مثل دیتاست)
        img_resized = cv2.resize(img, (100, 56))
        logger.info(f"Image resized to {img_resized.shape}")

        # فلت کردن تصویر
        img_flat = img_resized.flatten().reshape(1, -1)
        logger.info(f"Image flattened to {img_flat.shape}")

        # بررسی ابعاد
        expected_features = 100 * 56 * 3  # 16800
        if img_flat.shape[1] != expected_features:
            logger.error(f"Expected {expected_features} features, got {img_flat.shape[1]}")
            raise HTTPException(status_code=422, detail=f"Expected {expected_features} features, got {img_flat.shape[1]}")

        if pca_components.shape[0] != img_flat.shape[1]:
            logger.error(f"PCA components shape {pca_components.shape[0]} does not match image features {img_flat.shape[1]}")
            raise HTTPException(status_code=422, detail="PCA components shape mismatch")

        # پیش‌پردازش
        img_flat = img_flat / 255.0  # نرمال‌سازی اولیه
        logger.info("Applied initial normalization (division by 255)")

        img_scaled = scaler.transform(img_flat)  # استانداردسازی
        logger.info("Applied MinMaxScaler transformation")

        img_pca = np.dot(img_scaled - pca_mean, pca_components)  # اعمال PCA
        logger.info(f"Applied PCA, img_pca shape: {img_pca.shape}")

        # پیش‌بینی با LDA
        pred_idx = lda.predict(img_pca)[0]
        logger.info(f"Predicted class index: {pred_idx}")

        # نگاشت به برچسب
        label_map = {5: '5₺', 10: '10₺', 20: '20₺', 50: '50₺', 100: '100₺', 200: '200₺'}
        if pred_idx not in label_map:
            logger.error(f"Invalid prediction index: {pred_idx}, expected one of {list(label_map.keys())}")
            raise HTTPException(status_code=422, detail=f"Invalid prediction: {pred_idx}")

        prediction = label_map[pred_idx]
        logger.info(f"Prediction: {prediction}")

        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error processing image: {str(e)}")        

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
