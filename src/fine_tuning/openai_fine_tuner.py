"""
OpenAI Fine-tuning Manager for AI Pipeline
Handles dataset validation, fine-tuning job creation, and monitoring
"""

import json
import logging
import time
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
import openai
from dataclasses import dataclass

@dataclass
class FineTuningConfig:
    """Configuration for OpenAI fine-tuning"""
    model: str = "gpt-4o-mini"
    training_file: str = "data/processed/training_data.jsonl"
    validation_file: Optional[str] = None
    suffix: str = "ev-charging"
    hyperparameters: Optional[Dict[str, Any]] = None
    max_retries: int = 3
    wait_time: int = 60  # seconds between status checks

class OpenAIFineTuner:
    def __init__(self, openai_api_key: str, config: FineTuningConfig):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Fine-tuning job tracking
        self.current_job = None
        self.job_history = []
    
    def validate_dataset(self, training_file: str = None) -> Dict[str, Any]:
        """Validate training dataset format and content"""
        if not training_file:
            training_file = self.config.training_file
        
        self.logger.info(f"Validating dataset: {training_file}")
        
        try:
            # Check if file exists
            if not Path(training_file).exists():
                return {
                    'success': False,
                    'error': f'Training file not found: {training_file}'
                }
            
            # Validate JSONL format
            validation_result = self._validate_jsonl_format(training_file)
            if not validation_result['success']:
                return validation_result
            
            # Check file size and line count
            file_stats = self._get_file_stats(training_file)
            
            return {
                'success': True,
                'details': {
                    'file_path': training_file,
                    'file_size_mb': file_stats['size_mb'],
                    'total_lines': file_stats['total_lines'],
                    'valid_lines': file_stats['valid_lines'],
                    'format': 'OpenAI fine-tuning JSONL',
                    'model': self.config.model
                }
            }
            
        except Exception as e:
            self.logger.error(f"Dataset validation error: {e}")
            return {
                'success': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def _validate_jsonl_format(self, file_path: str) -> Dict[str, Any]:
        """Validate JSONL format and OpenAI fine-tuning schema"""
        try:
            valid_lines = 0
            total_lines = 0
            errors = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Check OpenAI fine-tuning format
                        if not isinstance(data, dict):
                            errors.append(f"Line {line_num}: Not a JSON object")
                            continue
                        
                        if 'messages' not in data:
                            errors.append(f"Line {line_num}: Missing 'messages' field")
                            continue
                        
                        messages = data['messages']
                        if not isinstance(messages, list) or len(messages) < 2:
                            errors.append(f"Line {line_num}: Invalid messages format")
                            continue
                        
                        # Check message structure
                        for msg in messages:
                            if not isinstance(msg, dict):
                                errors.append(f"Line {line_num}: Invalid message format")
                                continue
                            
                            if 'role' not in msg or 'content' not in msg:
                                errors.append(f"Line {line_num}: Message missing role or content")
                                continue
                            
                            if msg['role'] not in ['system', 'user', 'assistant']:
                                errors.append(f"Line {line_num}: Invalid role: {msg['role']}")
                                continue
                        
                        valid_lines += 1
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                        continue
            
            if errors:
                return {
                    'success': False,
                    'error': f'Format validation failed: {len(errors)} errors',
                    'details': errors[:5]  # Show first 5 errors
                }
            
            return {
                'success': True,
                'valid_lines': valid_lines,
                'total_lines': total_lines
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }
    
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get file statistics"""
        path = Path(file_path)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        return {
            'size_mb': round(size_mb, 2),
            'total_lines': total_lines,
            'valid_lines': total_lines  # Will be updated by validation
        }
    
    def create_fine_tuning_job(self, training_file: str = None, **kwargs) -> Dict[str, Any]:
        """Create a new fine-tuning job using OpenAI API"""
        if not training_file:
            training_file = self.config.training_file
        
        self.logger.info(f"Creating fine-tuning job for: {training_file}")
        
        try:
            # Validate dataset first
            validation = self.validate_dataset(training_file)
            if not validation['success']:
                return {
                    'success': False,
                    'error': f'Dataset validation failed: {validation["error"]}'
                }
            
            # Prepare job parameters
            job_params = {
                'model': self.config.model,
                'training_file': training_file,
                'suffix': kwargs.get('suffix', self.config.suffix)
            }
            
            # Add optional parameters
            if self.config.validation_file and Path(self.config.validation_file).exists():
                job_params['validation_file'] = self.config.validation_file
            
            if self.config.hyperparameters:
                job_params['hyperparameters'] = self.config.hyperparameters
            
            # Create fine-tuning job
            self.logger.info("Submitting fine-tuning job to OpenAI...")
            fine_tuning_job = self.openai_client.fine_tuning.jobs.create(**job_params)
            
            # Store job information
            self.current_job = {
                'id': fine_tuning_job.id,
                'status': fine_tuning_job.status,
                'created_at': fine_tuning_job.created_at,
                'model': fine_tuning_job.model,
                'training_file': fine_tuning_job.training_file,
                'suffix': fine_tuning_job.suffix
            }
            
            self.job_history.append(self.current_job.copy())
            
            self.logger.info(f"Fine-tuning job created successfully: {fine_tuning_job.id}")
            
            return {
                'success': True,
                'job_id': fine_tuning_job.id,
                'status': fine_tuning_job.status,
                'details': self.current_job
            }
            
        except Exception as e:
            error_msg = f"Error creating fine-tuning job: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_job_status(self, job_id: str = None) -> Dict[str, Any]:
        """Get the status of a fine-tuning job"""
        if not job_id and self.current_job:
            job_id = self.current_job['id']
        
        if not job_id:
            return {
                'success': False,
                'error': 'No job ID provided'
            }
        
        try:
            self.logger.info(f"Checking status for job: {job_id}")
            job = self.openai_client.fine_tuning.jobs.retrieve(job_id)
            
            # Update current job status
            if self.current_job and self.current_job['id'] == job_id:
                self.current_job['status'] = job.status
                self.current_job['finished_at'] = getattr(job, 'finished_at', None)
                self.current_job['fine_tuned_model'] = getattr(job, 'fine_tuned_model', None)
            
            return {
                'success': True,
                'job_id': job.id,
                'status': job.status,
                'created_at': job.created_at,
                'finished_at': getattr(job, 'finished_at', None),
                'fine_tuned_model': getattr(job, 'fine_tuned_model', None),
                'training_file': job.training_file,
                'validation_file': getattr(job, 'validation_file', None),
                'trained_tokens': getattr(job, 'trained_tokens', None),
                'error': getattr(job, 'error', None)
            }
            
        except Exception as e:
            error_msg = f"Error retrieving job status: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def list_jobs(self, limit: int = 10) -> Dict[str, Any]:
        """List recent fine-tuning jobs"""
        try:
            self.logger.info(f"Listing recent fine-tuning jobs (limit: {limit})")
            jobs = self.openai_client.fine_tuning.jobs.list(limit=limit)
            
            job_list = []
            for job in jobs.data:
                job_info = {
                    'id': job.id,
                    'status': job.status,
                    'model': job.model,
                    'created_at': job.created_at,
                    'finished_at': getattr(job, 'finished_at', None),
                    'fine_tuned_model': getattr(job, 'fine_tuned_model', None),
                    'suffix': job.suffix
                }
                job_list.append(job_info)
            
            return {
                'success': True,
                'jobs': job_list,
                'total': len(job_list)
            }
            
        except Exception as e:
            error_msg = f"Error listing jobs: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def monitor_job(self, job_id: str = None, max_wait_time: int = 3600) -> Dict[str, Any]:
        """Monitor a fine-tuning job until completion or timeout"""
        if not job_id and self.current_job:
            job_id = self.current_job['id']
        
        if not job_id:
            return {
                'success': False,
                'error': 'No job ID provided'
            }
        
        self.logger.info(f"Starting job monitoring for: {job_id}")
        start_time = time.time()
        
        while True:
            # Check if we've exceeded max wait time
            if time.time() - start_time > max_wait_time:
                return {
                    'success': False,
                    'error': f'Monitoring timeout after {max_wait_time} seconds',
                    'job_id': job_id
                }
            
            # Get current status
            status_result = self.get_job_status(job_id)
            if not status_result['success']:
                return status_result
            
            status = status_result['status']
            self.logger.info(f"Job {job_id} status: {status}")
            
            # Check for completion states
            if status in ['succeeded', 'failed', 'cancelled']:
                if status == 'succeeded':
                    self.logger.info(f"Fine-tuning job {job_id} completed successfully!")
                    return {
                        'success': True,
                        'status': status,
                        'job_id': job_id,
                        'fine_tuned_model': status_result.get('fine_tuned_model'),
                        'details': status_result
                    }
                else:
                    error_msg = status_result.get('error', 'Unknown error')
                    return {
                        'success': False,
                        'status': status,
                        'job_id': job_id,
                        'error': error_msg
                    }
            
            # Wait before next check
            self.logger.info(f"Waiting {self.config.wait_time} seconds before next status check...")
            time.sleep(self.config.wait_time)
    
    def cancel_job(self, job_id: str = None) -> Dict[str, Any]:
        """Cancel a running fine-tuning job"""
        if not job_id and self.current_job:
            job_id = self.current_job['id']
        
        if not job_id:
            return {
                'success': False,
                'error': 'No job ID provided'
            }
        
        try:
            self.logger.info(f"Cancelling job: {job_id}")
            job = self.openai_client.fine_tuning.jobs.cancel(job_id)
            
            return {
                'success': True,
                'job_id': job.id,
                'status': job.status,
                'message': 'Job cancelled successfully'
            }
            
        except Exception as e:
            error_msg = f"Error cancelling job: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_job_events(self, job_id: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get events/logs for a fine-tuning job"""
        if not job_id and self.current_job:
            job_id = self.current_job['id']
        
        if not job_id:
            return {
                'success': False,
                'error': 'No job ID provided'
            }
        
        try:
            self.logger.info(f"Retrieving events for job: {job_id}")
            events = self.openai_client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id,
                limit=limit
            )
            
            event_list = []
            for event in events.data:
                event_info = {
                    'id': event.id,
                    'created_at': event.created_at,
                    'level': event.level,
                    'message': event.message,
                    'type': event.type
                }
                event_list.append(event_info)
            
            return {
                'success': True,
                'job_id': job_id,
                'events': event_list,
                'total': len(event_list)
            }
            
        except Exception as e:
            error_msg = f"Error retrieving job events: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def save_job_history(self, output_file: str = "data/fine_tuning/job_history.json") -> bool:
        """Save job history to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.job_history, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Job history saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving job history: {e}")
            return False
