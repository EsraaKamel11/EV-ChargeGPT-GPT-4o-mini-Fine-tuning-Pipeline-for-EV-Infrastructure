"""
Model Registry for AI Pipeline
Handles model registration, versioning, and metadata tracking
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict

@dataclass
class ModelMetadata:
    """Metadata for a fine-tuned model"""
    model_id: str
    version: str
    base_model: str
    fine_tuned_model_name: str
    training_file: str
    training_data_size: int
    created_at: str
    status: str  # 'training', 'ready', 'deployed', 'archived'
    performance_metrics: Dict[str, Any]
    training_config: Dict[str, Any]
    description: str = ""
    tags: List[str] = None

class ModelRegistry:
    def __init__(self, registry_path: str = "data/model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Registry file
        self.registry_file = self.registry_path / "models.json"
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load existing model registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert back to ModelMetadata objects
                    models = {}
                    for model_id, model_data in data.items():
                        models[model_id] = ModelMetadata(**model_data)
                    return models
            except Exception as e:
                self.logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            # Convert ModelMetadata objects to dictionaries
            registry_data = {}
            for model_id, model in self.models.items():
                registry_data[model_id] = asdict(model)
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Registry saved to {self.registry_file}")
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def register_model(self, 
                      base_model: str,
                      fine_tuned_model_name: str,
                      training_file: str,
                      training_data_size: int,
                      training_config: Dict[str, Any],
                      description: str = "",
                      tags: List[str] = None) -> str:
        """Register a new fine-tuned model"""
        try:
            # Generate unique model ID
            timestamp = datetime.now().isoformat()
            model_id = f"ev-charging-{base_model}-{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
            
            # Create model metadata
            model_metadata = ModelMetadata(
                model_id=model_id,
                version="1.0.0",
                base_model=base_model,
                fine_tuned_model_name=fine_tuned_model_name,
                training_file=training_file,
                training_data_size=training_data_size,
                created_at=timestamp,
                status="ready",
                performance_metrics={},
                training_config=training_config,
                description=description,
                tags=tags or ["ev-charging", "fine-tuned"]
            )
            
            # Add to registry
            self.models[model_id] = model_metadata
            
            # Save registry
            self._save_registry()
            
            self.logger.info(f"Model registered: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def list_models(self, status: str = None, tags: List[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        filtered_models = []
        
        for model in self.models.values():
            # Filter by status
            if status and model.status != status:
                continue
            
            # Filter by tags
            if tags and not any(tag in model.tags for tag in tags):
                continue
            
            filtered_models.append(model)
        
        return filtered_models
    
    def update_model_status(self, model_id: str, status: str):
        """Update model status"""
        if model_id in self.models:
            self.models[model_id].status = status
            self._save_registry()
            self.logger.info(f"Model {model_id} status updated to: {status}")
        else:
            self.logger.warning(f"Model {model_id} not found in registry")
    
    def update_performance_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """Update model performance metrics"""
        if model_id in self.models:
            self.models[model_id].performance_metrics.update(metrics)
            self._save_registry()
            self.logger.info(f"Performance metrics updated for model {model_id}")
        else:
            self.logger.warning(f"Model {model_id} not found in registry")
    
    def archive_model(self, model_id: str):
        """Archive a model (mark as inactive)"""
        if model_id in self.models:
            self.models[model_id].status = "archived"
            self._save_registry()
            self.logger.info(f"Model {model_id} archived")
        else:
            self.logger.warning(f"Model {model_id} not found in registry")
    
    def get_latest_model(self, base_model: str = None) -> Optional[ModelMetadata]:
        """Get the latest model by creation date"""
        if not self.models:
            return None
        
        # Filter by base model if specified
        candidate_models = []
        for model in self.models.values():
            if base_model and model.base_model != base_model:
                continue
            if model.status in ["ready", "deployed"]:
                candidate_models.append(model)
        
        if not candidate_models:
            return None
        
        # Return the most recently created model
        return max(candidate_models, key=lambda x: x.created_at)
    
    def export_model_info(self, model_id: str, output_file: str = None) -> Dict[str, Any]:
        """Export complete model information"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        export_data = asdict(model)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Model info exported to {output_path}")
        
        return export_data
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the registry"""
        total_models = len(self.models)
        status_counts = {}
        base_model_counts = {}
        
        for model in self.models.values():
            # Count by status
            status_counts[model.status] = status_counts.get(model.status, 0) + 1
            
            # Count by base model
            base_model_counts[model.base_model] = base_model_counts.get(model.base_model, 0) + 1
        
        return {
            'total_models': total_models,
            'status_distribution': status_counts,
            'base_model_distribution': base_model_counts,
            'last_updated': datetime.now().isoformat()
        }
