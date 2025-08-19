"""Main FastAPI application for ML model serving."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.models import PredictionRequest, PredictionResponse, HealthResponse, ModelInfo
from src.api.middleware import RateLimitMiddleware, AuthenticationMiddleware, LoggingMiddleware
from src.api.simple_services import SimplePredictionService, SimpleModelService, SimpleMonitoringService
from src.api.exceptions import ModelNotFoundError, PredictionError, ValidationError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total predictions', ['model_name', 'version', 'status'])
PREDICTION_LATENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model_name', 'version'])
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active models')
ERROR_COUNTER = Counter('api_errors_total', 'Total API errors', ['error_type', 'endpoint'])


# Global services (initialized in lifespan)
prediction_service: Optional[SimplePredictionService] = None
model_service: Optional[SimpleModelService] = None
monitoring_service: Optional[SimpleMonitoringService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global prediction_service, model_service, monitoring_service
    
    logger.info("Starting up MLOps API server...")
    
    try:
        # Load configuration
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config = {}
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Initialize services
        prediction_service = SimplePredictionService(config)
        model_service = SimpleModelService(config)
        monitoring_service = SimpleMonitoringService(config)
        
        # Load models
        await prediction_service.load_models()
        
        # Update metrics
        active_models_count = len(prediction_service.get_loaded_models())
        ACTIVE_MODELS.set(active_models_count)
        
        logger.info(f"API server started successfully with {active_models_count} models loaded")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise
    finally:
        logger.info("Shutting down MLOps API server...")
        # Cleanup if needed
        if prediction_service:
            await prediction_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Production-ready API for ML model inference with monitoring and A/B testing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(LoggingMiddleware)


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    """Handle model not found errors."""
    ERROR_COUNTER.labels(error_type="model_not_found", endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "error_type": "model_not_found"}
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    """Handle prediction errors."""
    ERROR_COUNTER.labels(error_type="prediction_error", endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": "prediction_error"}
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    ERROR_COUNTER.labels(error_type="validation_error", endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "validation_error"}
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "MLOps Model Serving API", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are initialized
        if not all([prediction_service, model_service, monitoring_service]):
            return HealthResponse(
                status="unhealthy",
                timestamp=time.time(),
                details={"error": "Services not initialized"}
            )
        
        # Check model loading status
        loaded_models = prediction_service.get_loaded_models()
        
        # Check database connectivity
        db_healthy = await monitoring_service.check_database_health()
        
        # Check Redis connectivity
        redis_healthy = await monitoring_service.check_redis_health()
        
        status = "healthy" if db_healthy and redis_healthy and loaded_models else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            details={
                "loaded_models": len(loaded_models),
                "database_healthy": db_healthy,
                "redis_healthy": redis_healthy,
                "models": list(loaded_models.keys())
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            details={"error": str(e)}
        )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        if not prediction_service or not prediction_service.get_loaded_models():
            raise HTTPException(status_code=503, detail="No models loaded")
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(
    model_name: str,
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """Make predictions using specified model."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Get prediction from service
        result = await prediction_service.predict(
            model_name=model_name,
            features=request.features,
            user_id=user_id,
            request_id=request_id
        )
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNTER.labels(
            model_name=result.model_name,
            version=result.model_version,
            status="success"
        ).inc()
        
        PREDICTION_LATENCY.labels(
            model_name=result.model_name,
            version=result.model_version
        ).observe(latency)
        
        # Log prediction for monitoring (background task)
        background_tasks.add_task(
            monitoring_service.log_prediction,
            model_name=result.model_name,
            model_version=result.model_version,
            features=request.features,
            prediction=result.prediction,
            prediction_proba=result.prediction_proba,
            latency_ms=latency * 1000,
            user_id=user_id,
            request_id=request_id
        )
        
        return result
        
    except ModelNotFoundError:
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            version="unknown",
            status="error"
        ).inc()
        raise
        
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            version="unknown", 
            status="error"
        ).inc()
        
        logger.error(f"Prediction failed for model {model_name}: {e}")
        raise PredictionError(f"Prediction failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_default(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """Make predictions using default model."""
    # Use the default/primary model (e.g., churn-predictor)
    return await predict("churn-predictor", request, background_tasks, user_id)


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    try:
        models = await model_service.list_models()
        return models
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    try:
        model_info = await model_service.get_model_info(model_name)
        if not model_info:
            raise ModelNotFoundError(f"Model {model_name} not found")
        
        return model_info
        
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.post("/models/{model_name}/load")
async def load_model(model_name: str, version: Optional[str] = None):
    """Load or reload a model."""
    try:
        success = await model_service.load_model(model_name, version)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Update metrics
        active_models_count = len(prediction_service.get_loaded_models())
        ACTIVE_MODELS.set(active_models_count)
        
        return {"message": f"Model {model_name} loaded successfully", "version": version}
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a model from memory."""
    try:
        success = await model_service.unload_model(model_name)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found or already unloaded")
        
        # Update metrics
        active_models_count = len(prediction_service.get_loaded_models())
        ACTIVE_MODELS.set(active_models_count)
        
        return {"message": f"Model {model_name} unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.get("/models/{model_name}/health")
async def model_health(model_name: str):
    """Get health status for a specific model."""
    try:
        health = await monitoring_service.get_model_health(model_name)
        return health
        
    except Exception as e:
        logger.error(f"Failed to get model health for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model health")


@app.get("/monitoring/drift")
async def get_drift_status():
    """Get data drift monitoring status."""
    try:
        drift_status = await monitoring_service.get_drift_summary()
        return drift_status
        
    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get drift status")


@app.get("/monitoring/performance")
async def get_performance_metrics():
    """Get model performance metrics."""
    try:
        performance = await monitoring_service.get_performance_summary()
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@app.get("/monitoring/alerts")
async def get_active_alerts():
    """Get active monitoring alerts."""
    try:
        alerts = await monitoring_service.get_active_alerts()
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ab-test/{experiment_name}/status")
async def ab_test_status(experiment_name: str):
    """Get A/B test status."""
    try:
        status = await prediction_service.get_ab_test_status(experiment_name)
        return status
        
    except Exception as e:
        logger.error(f"Failed to get A/B test status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get A/B test status")


@app.post("/ab-test/setup")
async def setup_ab_test(
    model_a_name: str,
    model_a_version: str,
    model_b_name: str, 
    model_b_version: str,
    traffic_split: float = 0.5,
    experiment_name: Optional[str] = None
):
    """Setup A/B test between two models."""
    try:
        result = await prediction_service.setup_ab_test(
            model_a_name=model_a_name,
            model_a_version=model_a_version,
            model_b_name=model_b_name,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            experiment_name=experiment_name
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to setup A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to setup A/B test: {str(e)}")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with authentication."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Interactive Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


def custom_openapi():
    """Custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="MLOps Model Serving API",
        version="1.0.0",
        description="Production-ready API for ML model inference with monitoring and A/B testing",
        routes=app.routes,
    )
    
    # Add authentication scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add global security requirement
    openapi_schema["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


async def run_server():
    """Run the FastAPI server."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False,  # Set to True for development
        workers=1  # Use multiple workers for production
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(run_server())