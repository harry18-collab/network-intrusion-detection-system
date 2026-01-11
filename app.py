import sys
import os
from contextlib import asynccontextmanager

import certifi
ca = certifi.where()

from dotenv import load_dotenv  # type: ignore
load_dotenv()

from networksecurity.logging.logger import logging  # type: ignore
from networksecurity.exception.exception import NetworkSecurityException  # type: ignore

mongo_db_url = os.getenv("MONGODB_URL_KEY")
if mongo_db_url:
    logging.info("MongoDB connection configured")
else:
    logging.warning("MongoDB URL not found in environment variables")
import pymongo
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from uvicorn import run as app_run
from fastapi.responses import Response, FileResponse
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Global MongoDB client
client = None
database = None
collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, database, collection
    from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
    from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
    
    client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]
    yield
    if client:
        client.close()

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS with specific origins for security
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/predict-ui")
async def predict_ui(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/train-ui")
async def train_ui(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.get("/prediction_output/output.csv")
async def download_results():
    return FileResponse("prediction_output/output.csv", filename="prediction_results.csv")

@app.get("/train")
async def train_route():
    print("\nStarting model training...")
    logging.info("Training pipeline initiated via API")
    
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
        # Check if model files were created successfully
        model_exists = os.path.exists("final_model/model.pkl")
        preprocessor_exists = os.path.exists("final_model/preprocessor.pkl")
        
        if model_exists and preprocessor_exists:
            print("Training completed successfully!")
            print("Model is ready for predictions.")
            logging.info("Training pipeline completed successfully")
            return {"status": "success", "message": "Training completed successfully"}
        else:
            print("Training completed but model files were not created.")
            return {"status": "error", "message": "Training completed but model files not created"}
            
    except Exception as e:
        print(f"Training failed: {str(e)}")
        logging.error(f"Training failed: {str(e)}")
        return {"status": "error", "message": f"Training failed: {str(e)}"}

@app.get("/training-status")
async def get_training_status():
    # Check if model files exist to determine training status
    model_exists = os.path.exists("final_model/model.pkl")
    preprocessor_exists = os.path.exists("final_model/preprocessor.pkl")
    
    if model_exists and preprocessor_exists:
        return {"status": "completed", "message": "Model is trained and ready"}
    else:
        return {"status": "not_trained", "message": "Model needs training"}


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Validate file size (10MB limit)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large")
        
        df = pd.read_csv(file.file)
        
        # Validate DataFrame is not empty
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Add target column if missing (for preprocessing compatibility)
        if 'Statistical_report' not in df.columns:
            df['Statistical_report'] = 0  # dummy values
        
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        print(f"Processing {len(df)} samples for prediction...")
        logging.info(f"Processing prediction for {len(df)} rows")
        y_pred = network_model.predict(df)
        
        # Calculate summary statistics
        total_samples = len(y_pred)
        safe_count = int((y_pred == 0).sum())
        threat_count = int((y_pred == 1).sum())
        safe_percentage = round((safe_count / total_samples) * 100, 1)
        threat_percentage = round((threat_count / total_samples) * 100, 1)
        
        # Remove the dummy target column before adding predictions
        if 'Statistical_report' in df.columns:
            df = df.drop('Statistical_report', axis=1)
        df['predicted_column'] = y_pred
        
        # Ensure output directory exists
        os.makedirs('prediction_output', exist_ok=True)
        df.to_csv('prediction_output/output.csv', index=False)
        
        # Generate HTML table with proper escaping
        table_html = df.to_html(classes='table table-striped table-hover', escape=True, table_id='prediction-table')
        
        return templates.TemplateResponse("table.html", {
            "request": request, 
            "table": table_html,
            "total_samples": total_samples,
            "safe_count": safe_count,
            "threat_count": threat_count,
            "safe_percentage": safe_percentage,
            "threat_percentage": threat_percentage
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise NetworkSecurityException(e, sys)  # type: ignore

    
if __name__=="__main__":
    app_run(app,host="127.0.0.1",port=8002)
