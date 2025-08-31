"""
FastAPI Inference Server for Fine-tuned Models
Provides API endpoints for model predictions with authentication and monitoring
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

import openai
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .model_registry import ModelRegistry, ModelMetadata

# Security
security = HTTPBearer()

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    question: str = Field(..., description="EV charging related question")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    max_tokens: int = Field(500, description="Maximum tokens in response")
    temperature: float = Field(0.7, description="Response creativity (0-1)")
    
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    answer: str
    model_id: str
    model_name: str
    response_time: float
    tokens_used: int
    confidence_score: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_available: int
    uptime: float

class ModelInfo(BaseModel):
    """Model information response"""
    model_id: str
    version: str
    base_model: str
    status: str
    performance_metrics: Dict[str, Any]
    created_at: str
    description: str

class InferenceServer:
    def __init__(self, openai_api_key: str, registry_path: str = "data/model_registry"):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model_registry = ModelRegistry(registry_path)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0
        self.error_count = 0
        
        # Create FastAPI app
        self.app = self._create_app()
        self._setup_routes()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title="EV Charging AI Inference Server",
            description="API for fine-tuned EV charging language models",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        return app
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint"""
            return {
                "message": "EV Charging AI Inference Server",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            uptime = time.time() - self.start_time
            models_available = len(self.model_registry.list_models(status="deployed"))
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                models_available=models_available,
                uptime=uptime
            )
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models(
            status: Optional[str] = None,
            tags: Optional[List[str]] = None
        ):
            """List available models"""
            try:
                models = self.model_registry.list_models(status=status, tags=tags)
                return [
                    ModelInfo(
                        model_id=model.model_id,
                        version=model.version,
                        base_model=model.base_model,
                        status=model.status,
                        performance_metrics=model.performance_metrics,
                        created_at=model.created_at,
                        description=model.description
                    )
                    for model in models
                ]
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models/{model_id}", response_model=ModelInfo)
        async def get_model_info(model_id: str):
            """Get specific model information"""
            try:
                model = self.model_registry.get_model(model_id)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                
                return ModelInfo(
                    model_id=model.model_id,
                    version=model.version,
                    base_model=model.base_model,
                    status=model.status,
                    performance_metrics=model.performance_metrics,
                    created_at=model.created_at,
                    description=model.description
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting model info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            background_tasks: BackgroundTasks = None
        ):
            """Make a prediction using the fine-tuned model"""
            try:
                # Validate API key (simple check - enhance for production)
                if not self._validate_api_key(credentials.credentials):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                # Start timing
                start_time = time.time()
                
                # Get model to use
                model_to_use = self._get_model_for_prediction(request.model_id)
                if not model_to_use:
                    raise HTTPException(status_code=400, detail="No suitable model available")
                
                # Make prediction
                prediction_result = await self._make_prediction(
                    request.question,
                    model_to_use.fine_tuned_model_name,
                    request.max_tokens,
                    request.temperature
                )
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Update metrics
                self._update_metrics(response_time)
                
                # Background task to log request
                if background_tasks:
                    background_tasks.add_task(
                        self._log_prediction_request,
                        request.question,
                        model_to_use.model_id,
                        response_time,
                        prediction_result
                    )
                
                return PredictionResponse(
                    answer=prediction_result['answer'],
                    model_id=model_to_use.model_id,
                    model_name=model_to_use.fine_tuned_model_name,
                    response_time=response_time,
                    tokens_used=prediction_result.get('tokens_used', 0),
                    confidence_score=prediction_result.get('confidence_score', 0.8),
                    timestamp=datetime.now().isoformat()
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                self.error_count += 1
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get server performance metrics"""
            avg_response_time = (
                self.total_response_time / self.request_count 
                if self.request_count > 0 else 0
            )
            
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "success_rate": (
                    (self.request_count - self.error_count) / self.request_count * 100
                    if self.request_count > 0 else 100
                ),
                "average_response_time": avg_response_time,
                "uptime": time.time() - self.start_time,
                "models_available": len(self.model_registry.list_models(status="deployed"))
            }
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key (implement proper validation for production)"""
        # For now, accept any non-empty key
        # In production, implement proper key validation
        return bool(api_key and len(api_key) > 0)
    
    def _get_model_for_prediction(self, requested_model_id: str = None) -> Optional[ModelMetadata]:
        """Get the model to use for prediction"""
        if requested_model_id:
            # Use specific model if requested
            model = self.model_registry.get_model(requested_model_id)
            if model and model.status == "deployed":
                return model
            return None
        
        # Get latest deployed model
        return self.model_registry.get_latest_model()
    
    async def _make_prediction(self, 
                              question: str, 
                              model_name: str, 
                              max_tokens: int, 
                              temperature: float) -> Dict[str, Any]:
        """Make prediction using OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert on electric vehicle charging infrastructure. Provide accurate, helpful, and detailed information."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            usage = response.usage
            
            return {
                'answer': answer,
                'tokens_used': usage.total_tokens if usage else 0,
                'confidence_score': 0.9  # Placeholder - implement proper confidence scoring
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def _update_metrics(self, response_time: float):
        """Update performance metrics"""
        self.request_count += 1
        self.total_response_time += response_time
    
    async def _log_prediction_request(self, 
                                    question: str, 
                                    model_id: str, 
                                    response_time: float, 
                                    result: Dict[str, Any]):
        """Log prediction request for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'model_id': model_id,
            'response_time': response_time,
            'tokens_used': result.get('tokens_used', 0),
            'success': True
        }
        
        # Save to log file
        log_file = Path("logs/inference_requests.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Error logging request: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the inference server"""
        self.logger.info(f"Starting inference server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if not debug else "debug"
        )

def create_inference_server(openai_api_key: str, registry_path: str = "data/model_registry") -> InferenceServer:
    """Factory function to create inference server"""
    return InferenceServer(openai_api_key, registry_path)
