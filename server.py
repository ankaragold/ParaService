from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
from io import BytesIO

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# لود مدل‌ها
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca_components.pkl', 'rb') as f:
    pca_components = pickle.load(f)
with open('pca_mean.pkl', 'rb') as f:
    pca_mean = pickle.load(f)
with open('lda.pkl', 'rb') as f:
    lda = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # خواندن تصویر
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # تصویر رنگی (RGB)
    
    # پیش‌پردازش (رزولوشن 100x56)
    img_resized = cv2.resize(img, (100, 56))
    img_flat = img_resized.flatten().reshape(1, -1)  # 100x56x3 = 16800
    
    # اعمال نرمال‌سازی، PCA و LDA
    img_scaled = scaler.transform(img_flat)
    img_pca = np.dot(img_scaled - pca_mean, pca_components)
    pred = lda.predict(img_pca)[0]
    
    # نگاشت به کلاس
    classes = ['5₺', '10₺', '20₺', '50₺', '100₺', '200₺']
    return {"prediction": classes[int(pred)]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
