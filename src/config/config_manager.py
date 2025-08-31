"""
Configuration Manager for AI Pipeline
Handles YAML configuration and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.env_vars = {}
        
        # Load environment variables
        load_dotenv()
        self._load_environment_variables()
        
        # Load YAML configuration
        self._load_yaml_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_environment_variables(self):
        """Load all relevant environment variables"""
        self.env_vars = {
            'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite:///./data/ai_pipeline.db'),
            'VECTOR_DB_PATH': os.getenv('VECTOR_DB_PATH', './data/vector_db'),
            'MODEL_CACHE_DIR': os.getenv('MODEL_CACHE_DIR', './models'),
            'OUTPUT_DIR': os.getenv('OUTPUT_DIR', './outputs'),
            'DATA_DIR': os.getenv('DATA_DIR', './data'),
            'SCRAPING_DELAY': float(os.getenv('SCRAPING_DELAY', '2.0')),
            'MAX_CONCURRENT_REQUESTS': int(os.getenv('MAX_CONCURRENT_REQUESTS', '5')),
            'REQUEST_TIMEOUT': int(os.getenv('REQUEST_TIMEOUT', '30')),
            'WANDB_PROJECT': os.getenv('WANDB_PROJECT', 'ev-charging-llm'),
            'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'LOG_FILE': os.getenv('LOG_FILE', './logs/pipeline.log'),
            'SECRET_KEY': os.getenv('SECRET_KEY', 'default-secret-key'),
            'ALLOWED_HOSTS': os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            'ENABLE_METRICS': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'METRICS_PORT': int(os.getenv('METRICS_PORT', '9090'))
        }
    
    def _load_yaml_config(self):
        """Load YAML configuration file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    self.config = yaml.safe_load(file)
                logging.info(f"Configuration loaded from {self.config_path}")
            else:
                logging.warning(f"Configuration file not found: {self.config_path}")
                self.config = {}
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            self.config = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.env_vars['LOG_LEVEL'].upper())
        log_file = Path(self.env_vars['LOG_FILE'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking YAML first, then environment variables"""
        # Check YAML config first
        if key in self.config:
            return self.config[key]
        
        # Check environment variables
        if key.upper() in self.env_vars:
            return self.env_vars[key.upper()]
        
        return default
    
    def get_target_domain(self) -> Dict[str, Any]:
        """Get target domain configuration"""
        return self.config.get('target_domain', {})
    
    def get_data_collection_config(self) -> Dict[str, Any]:
        """Get data collection configuration"""
        return self.config.get('data_collection', {})
    
    def get_fine_tuning_config(self) -> Dict[str, Any]:
        """Get fine-tuning configuration"""
        return self.config.get('fine_tuning', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration"""
        return self.config.get('deployment', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('fine_tuning', {}).get('model_config', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('fine_tuning', {}).get('training', {})
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration"""
        return self.config.get('fine_tuning', {}).get('quantization', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config.update(updates)
        
        # Save updated config to file
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            logging.info("Configuration updated and saved")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def create_directories(self):
        """Create necessary directories for the pipeline"""
        directories = [
            self.env_vars['DATA_DIR'],
            self.env_vars['OUTPUT_DIR'],
            self.env_vars['MODEL_CACHE_DIR'],
            self.env_vars['VECTOR_DB_PATH'],
            Path(self.env_vars['LOG_FILE']).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory created/verified: {directory}")
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_sections = ['target_domain', 'data_collection', 'fine_tuning', 'evaluation', 'deployment']
        
        for section in required_sections:
            if section not in self.config:
                logging.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate target domain
        target_domain = self.config['target_domain']
        if not target_domain.get('name') or not target_domain.get('use_case'):
            logging.error("Target domain must have name and use_case")
            return False
        
        # Validate fine-tuning config
        fine_tuning = self.config['fine_tuning']
        if not fine_tuning.get('base_model'):
            logging.error("Fine-tuning must specify base_model")
            return False
        
        logging.info("Configuration validation passed")
        return True
