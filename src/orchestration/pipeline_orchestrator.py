"""
Pipeline Orchestrator for AI Pipeline
Handles workflow automation, scheduling, and pipeline execution
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
import schedule
import threading
from dataclasses import dataclass
from enum import Enum

from ..config.config_manager import ConfigManager
from ..data_collection.web_scraper import WebScraper
from ..data_collection.pdf_extractor import PDFExtractor
from ..data_processing.data_processor import DataProcessor, ProcessingConfig
from ..fine_tuning.openai_fine_tuner import OpenAIFineTuner, FineTuningConfig
from ..evaluation.model_evaluator import ModelEvaluator, EvaluationConfig
from ..deployment.model_registry import ModelRegistry

class PipelineStatus(Enum):
    """Pipeline execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineExecution:
    """Pipeline execution record"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.IDLE
    stages_completed: List[str] = None
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    results: Dict[str, Any] = None

class PipelineOrchestrator:
    def __init__(self, config_manager: ConfigManager, openai_api_key: str):
        self.config_manager = config_manager
        self.openai_api_key = openai_api_key
        self.logger = logging.getLogger(__name__)
        
        # Pipeline components
        self.web_scraper = None
        self.pdf_extractor = None
        self.data_processor = None
        self.fine_tuner = None
        self.evaluator = None
        self.model_registry = ModelRegistry()
        
        # Execution tracking
        self.executions: Dict[str, PipelineExecution] = {}
        self.current_execution: Optional[PipelineExecution] = None
        
        # Callbacks
        self.on_stage_complete: Optional[Callable] = None
        self.on_pipeline_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Scheduling
        self.scheduler = schedule.Scheduler()
        self.scheduler_thread = None
        self.running = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components"""
        try:
            # Initialize data collection components
            web_config = self.config_manager.get_data_collection_config().get('web_scraping', {})
            if web_config.get('enabled', False):
                self.web_scraper = WebScraper(web_config)
            
            pdf_config = self.config_manager.get_data_collection_config().get('pdf_extraction', {})
            if pdf_config.get('enabled', False):
                self.pdf_extractor = PDFExtractor(pdf_config)
            
            # Initialize data processing
            processing_config = ProcessingConfig()
            self.data_processor = DataProcessor(processing_config, self.openai_api_key)
            
            # Initialize fine-tuning
            fine_tuning_config = FineTuningConfig()
            self.fine_tuner = OpenAIFineTuner(self.openai_api_key, fine_tuning_config)
            
            # Initialize evaluation
            evaluation_config = EvaluationConfig()
            self.evaluator = ModelEvaluator(self.openai_api_key, evaluation_config)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def execute_pipeline(self, 
                             execution_id: str = None,
                             stages: List[str] = None,
                             force_rerun: bool = False) -> str:
        """Execute the complete AI pipeline"""
        if execution_id is None:
            execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if pipeline is already running
        if self.current_execution and self.current_execution.status == PipelineStatus.RUNNING:
            raise RuntimeError("Pipeline is already running")
        
        # Create execution record
        execution = PipelineExecution(
            execution_id=execution_id,
            start_time=datetime.now(),
            status=PipelineStatus.RUNNING,
            stages_completed=[],
            results={}
        )
        
        self.executions[execution_id] = execution
        self.current_execution = execution
        
        self.logger.info(f"Starting pipeline execution: {execution_id}")
        
        try:
            # Define pipeline stages
            if stages is None:
                stages = [
                    "data_collection",
                    "data_processing", 
                    "fine_tuning",
                    "evaluation"
                ]
            
            execution.current_stage = "initializing"
            
            # Execute each stage
            for stage in stages:
                if execution.status == PipelineStatus.CANCELLED:
                    break
                
                execution.current_stage = stage
                self.logger.info(f"Executing stage: {stage}")
                
                try:
                    stage_result = await self._execute_stage(stage, force_rerun)
                    execution.results[stage] = stage_result
                    execution.stages_completed.append(stage)
                    
                    # Call stage completion callback
                    if self.on_stage_complete:
                        self.on_stage_complete(stage, stage_result)
                    
                    self.logger.info(f"Stage {stage} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Stage {stage} failed: {str(e)}"
                    self.logger.error(error_msg)
                    execution.error_message = error_msg
                    execution.status = PipelineStatus.FAILED
                    
                    # Call error callback
                    if self.on_error:
                        self.on_error(stage, e)
                    
                    break
            
            # Update execution status
            if execution.status == PipelineStatus.RUNNING:
                execution.status = PipelineStatus.COMPLETED
                execution.end_time = datetime.now()
                
                # Call completion callback
                if self.on_pipeline_complete:
                    self.on_pipeline_complete(execution)
                
                self.logger.info(f"Pipeline execution {execution_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            execution.error_message = error_msg
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.now()
            
            # Call error callback
            if self.on_error:
                self.on_error("pipeline", e)
        
        finally:
            self.current_execution = None
        
        return execution_id
    
    async def _execute_stage(self, stage: str, force_rerun: bool = False) -> Dict[str, Any]:
        """Execute a specific pipeline stage"""
        if stage == "data_collection":
            return await self._execute_data_collection()
        elif stage == "data_processing":
            return await self._execute_data_processing()
        elif stage == "fine_tuning":
            return await self._execute_fine_tuning()
        elif stage == "evaluation":
            return await self._execute_evaluation()
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    async def _execute_data_collection(self) -> Dict[str, Any]:
        """Execute data collection stage"""
        results = {
            'web_data': None,
            'pdf_data': None,
            'total_items': 0
        }
        
        # Web scraping
        if self.web_scraper:
            try:
                web_data = await self.web_scraper.scrape_ev_websites()
                if web_data:
                    self.web_scraper.save_scraped_data("data/collected/web_data.json")
                    results['web_data'] = len(web_data)
                    results['total_items'] += len(web_data)
            except Exception as e:
                self.logger.warning(f"Web scraping failed: {e}")
        
        # PDF extraction
        if self.pdf_extractor:
            try:
                pdf_dir = Path("data/pdfs")
                if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
                    pdf_data = self.pdf_extractor.process_pdf_directory(str(pdf_dir))
                    if pdf_data:
                        self.pdf_extractor.save_extracted_data("data/collected/pdf_data.json")
                        results['pdf_data'] = len(pdf_data)
                        results['total_items'] += len(pdf_data)
            except Exception as e:
                self.logger.warning(f"PDF extraction failed: {e}")
        
        return results
    
    async def _execute_data_processing(self) -> Dict[str, Any]:
        """Execute data processing stage"""
        # Load collected data
        collected_data = []
        
        web_file = Path("data/collected/web_data.json")
        if web_file.exists():
            try:
                with open(web_file, 'r', encoding='utf-8') as f:
                    web_data = json.load(f)
                    collected_data.extend(web_data)
            except Exception as e:
                self.logger.warning(f"Error loading web data: {e}")
        
        pdf_file = Path("data/collected/pdf_data.json")
        if pdf_file.exists():
            try:
                with open(pdf_file, 'r', encoding='utf-8') as f:
                    pdf_data = json.load(f)
                    collected_data.extend(pdf_data)
            except Exception as e:
                self.logger.warning(f"Error loading PDF data: {e}")
        
        if not collected_data:
            raise RuntimeError("No data available for processing")
        
        # Process data
        training_data = self.data_processor.process_collected_data(collected_data)
        output_files = self.data_processor.save_processed_data("data/processed")
        
        return {
            'input_items': len(collected_data),
            'training_examples': len(training_data),
            'output_files': output_files
        }
    
    async def _execute_fine_tuning(self) -> Dict[str, Any]:
        """Execute fine-tuning stage"""
        training_file = "data/processed/training_data.jsonl"
        
        if not Path(training_file).exists():
            raise RuntimeError("Training data not found")
        
        # Create fine-tuning job
        job_result = self.fine_tuner.create_fine_tuning_job(training_file)
        
        if not job_result['success']:
            raise RuntimeError(f"Fine-tuning job creation failed: {job_result['error']}")
        
        # Register model in registry
        model_id = self.model_registry.register_model(
            base_model=self.fine_tuner.config.model,
            fine_tuned_model_name=job_result['job_id'],
            training_file=training_file,
            training_data_size=job_result.get('training_data_size', 0),
            training_config={
                'model': self.fine_tuner.config.model,
                'suffix': self.fine_tuner.config.suffix
            },
            description="EV charging domain fine-tuned model"
        )
        
        return {
            'job_id': job_result['job_id'],
            'model_id': model_id,
            'status': job_result['status']
        }
    
    async def _execute_evaluation(self) -> Dict[str, Any]:
        """Execute evaluation stage"""
        # Create benchmark
        benchmark_result = self.evaluator.create_domain_benchmark()
        
        if not benchmark_result['success']:
            raise RuntimeError(f"Benchmark creation failed: {benchmark_result['error']}")
        
        # Evaluate models
        evaluation_result = self.evaluator.evaluate_models()
        
        if not evaluation_result['success']:
            raise RuntimeError(f"Model evaluation failed: {evaluation_result['error']}")
        
        # Save results
        self.evaluator.save_evaluation_results()
        report = self.evaluator.generate_evaluation_report()
        
        return {
            'benchmark_items': evaluation_result['benchmark_items'],
            'baseline_model': evaluation_result['baseline_model'],
            'fine_tuned_model': evaluation_result['fine_tuned_model'],
            'report_file': report
        }
    
    def cancel_pipeline(self, execution_id: str = None):
        """Cancel a pipeline execution"""
        if execution_id is None and self.current_execution:
            execution_id = self.current_execution.execution_id
        
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            execution.status = PipelineStatus.CANCELLED
            execution.end_time = datetime.now()
            self.logger.info(f"Pipeline execution {execution_id} cancelled")
        else:
            self.logger.warning(f"Pipeline execution {execution_id} not found")
    
    def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def list_executions(self, status: PipelineStatus = None) -> List[PipelineExecution]:
        """List pipeline executions"""
        if status is None:
            return list(self.executions.values())
        
        return [exec for exec in self.executions.values() if exec.status == status]
    
    def schedule_pipeline(self, 
                         schedule_time: str, 
                         stages: List[str] = None,
                         recurring: bool = True):
        """Schedule pipeline execution"""
        def run_pipeline():
            asyncio.run(self.execute_pipeline(stages=stages))
        
        if recurring:
            self.scheduler.every().day.at(schedule_time).do(run_pipeline)
        else:
            self.scheduler.every().day.at(schedule_time).do(run_pipeline).tag('one_time')
        
        self.logger.info(f"Pipeline scheduled for {schedule_time}")
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            return
        
        def run_scheduler():
            while self.running:
                self.scheduler.run_pending()
                time.sleep(1)
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Pipeline scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Pipeline scheduler stopped")
    
    def set_callbacks(self, 
                     on_stage_complete: Callable = None,
                     on_pipeline_complete: Callable = None,
                     on_error: Callable = None):
        """Set pipeline callbacks"""
        self.on_stage_complete = on_stage_complete
        self.on_pipeline_complete = on_pipeline_complete
        self.on_error = on_error
    
    def export_execution_log(self, execution_id: str, output_file: str = None) -> Dict[str, Any]:
        """Export execution log"""
        execution = self.get_execution_status(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        log_data = {
            'execution_id': execution.execution_id,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'status': execution.status.value,
            'stages_completed': execution.stages_completed,
            'current_stage': execution.current_stage,
            'error_message': execution.error_message,
            'results': execution.results
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Execution log exported to {output_path}")
        
        return log_data
