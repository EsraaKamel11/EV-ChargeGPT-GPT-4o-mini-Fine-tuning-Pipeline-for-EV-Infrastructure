"""
Data Processor for AI Pipeline
Handles data cleaning, deduplication, quality filtering, and Q-A generation
"""

import json
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import openai
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    min_text_length: int = 100
    max_text_length: int = 5000
    min_ev_keywords: int = 2
    max_duplicate_similarity: float = 0.9
    qa_generation_batch_size: int = 10
    max_qa_pairs_per_text: int = 3
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 500
    openai_temperature: float = 0.7

class DataProcessor:
    def __init__(self, config: ProcessingConfig, openai_api_key: str):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Processing results
        self.cleaned_data = []
        self.qa_pairs = []
        self.processing_stats = {}
        
    def process_collected_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Main processing pipeline"""
        self.logger.info(f"Starting data processing for {len(data)} items")
        
        # Step 1: Clean and filter data
        cleaned_data = self._clean_and_filter_data(data)
        self.logger.info(f"Cleaned data: {len(cleaned_data)} items")
        
        # Step 2: Remove duplicates
        deduplicated_data = self._remove_duplicates(cleaned_data)
        self.logger.info(f"Deduplicated data: {len(deduplicated_data)} items")
        
        # Step 3: Generate Q-A pairs
        qa_pairs = self._generate_qa_pairs(deduplicated_data)
        self.logger.info(f"Generated {len(qa_pairs)} Q-A pairs")
        
        # Step 4: Format for OpenAI fine-tuning
        training_data = self._format_for_fine_tuning(qa_pairs)
        
        self.cleaned_data = cleaned_data
        self.qa_pairs = qa_pairs
        
        return training_data
    
    def _clean_and_filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and filter collected data"""
        cleaned_data = []
        
        for item in tqdm(data, desc="Cleaning data"):
            try:
                # Extract text content
                text = self._extract_text_content(item)
                if not text:
                    continue
                
                # Clean text
                cleaned_text = self._clean_text(text)
                if not cleaned_text:
                    continue
                
                # Quality filtering
                if not self._passes_quality_filter(item, cleaned_text):
                    continue
                
                # Create cleaned item
                cleaned_item = {
                    'id': self._generate_id(item),
                    'source_type': item.get('domain', 'pdf'),
                    'source_url': item.get('url', item.get('file_path', '')),
                    'cleaned_text': cleaned_text,
                    'ev_keywords': item.get('ev_keywords', []),
                    'metadata': {
                        'original_length': len(text),
                        'cleaned_length': len(cleaned_text),
                        'processing_timestamp': time.time()
                    }
                }
                
                cleaned_data.append(cleaned_item)
                
            except Exception as e:
                self.logger.warning(f"Error processing item: {e}")
                continue
        
        return cleaned_data
    
    def _extract_text_content(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from different data types"""
        if 'text' in item:
            return item['text']
        elif 'text_content' in item and 'full_text' in item['text_content']:
            return item['text_content']['full_text']
        return None
    
    def _clean_text(self, text: str) -> Optional[str]:
        """Clean and normalize text"""
        if not text or len(text.strip()) < self.config.min_text_length:
            return None
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove very long text
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        # Basic text cleaning
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text if len(text) >= self.config.min_text_length else None
    
    def _passes_quality_filter(self, item: Dict[str, Any], text: str) -> bool:
        """Check if item passes quality filters"""
        # Check text length
        if len(text) < self.config.min_text_length:
            return False
        
        # Check EV keyword presence
        ev_keywords = item.get('ev_keywords', [])
        if len(ev_keywords) < self.config.min_ev_keywords:
            return False
        
        # Check for meaningful content (not just repeated words)
        words = text.split()
        if len(set(words)) < len(words) * 0.3:  # At least 30% unique words
            return False
        
        return True
    
    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and very similar content"""
        if not data:
            return []
        
        # Simple deduplication based on text similarity
        unique_data = []
        seen_hashes = set()
        
        for item in tqdm(data, desc="Removing duplicates"):
            text_hash = hashlib.md5(item['cleaned_text'].encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_data.append(item)
        
        self.logger.info(f"Removed {len(data) - len(unique_data)} duplicates")
        return unique_data
    
    def _generate_qa_pairs(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate Q-A pairs using GPT-4o-mini"""
        qa_pairs = []
        
        # Process in batches
        for i in range(0, len(data), self.config.qa_generation_batch_size):
            batch = data[i:i + self.config.qa_generation_batch_size]
            
            batch_qa = self._generate_qa_batch(batch)
            qa_pairs.extend(batch_qa)
            
            # Rate limiting
            time.sleep(1)
        
        return qa_pairs
    
    def _generate_qa_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate Q-A pairs for a batch of texts"""
        qa_pairs = []
        
        for item in batch:
            try:
                # Generate Q-A pairs for this text
                text_qa_pairs = self._generate_qa_for_text(
                    item['cleaned_text'], 
                    item['ev_keywords']
                )
                
                for qa_pair in text_qa_pairs:
                    qa_pairs.append({
                        'id': f"{item['id']}_qa_{len(qa_pairs)}",
                        'source_id': item['id'],
                        'question': qa_pair['question'],
                        'answer': qa_pair['answer'],
                        'ev_keywords': qa_pair['ev_keywords'],
                        'metadata': item['metadata']
                    })
                
            except Exception as e:
                self.logger.warning(f"Error generating Q-A for item {item['id']}: {e}")
                continue
        
        return qa_pairs
    
    def _generate_qa_for_text(self, text: str, ev_keywords: List[str]) -> List[Dict[str, Any]]:
        """Generate Q-A pairs for a single text using GPT-4o-mini"""
        try:
            # Create prompt for Q-A generation
            prompt = self._create_qa_generation_prompt(text, ev_keywords)
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert on electric vehicle charging infrastructure. Generate realistic questions and answers based on the provided text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.openai_max_tokens,
                temperature=self.config.openai_temperature,
                n=min(self.config.max_qa_pairs_per_text, 3)
            )
            
            # Parse response
            qa_pairs = self._parse_qa_response(response.choices[0].message.content)
            
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            # Fallback: generate simple Q-A pairs
            return self._generate_fallback_qa(text, ev_keywords)
    
    def _create_qa_generation_prompt(self, text: str, ev_keywords: List[str]) -> str:
        """Create prompt for Q-A generation"""
        return f"""
Based on the following text about electric vehicle charging, generate {self.config.max_qa_pairs_per_text} realistic question-answer pairs.

Text: {text[:1000]}...

EV Keywords: {', '.join(ev_keywords[:5])}

Generate the Q-A pairs in this format:
Q1: [Question about EV charging]
A1: [Answer based on the text]

Q2: [Another question]
A2: [Another answer]

Make sure questions are practical and answers are informative and accurate.
"""
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse Q-A response from OpenAI"""
        qa_pairs = []
        
        # Simple parsing - look for Q: and A: patterns
        lines = response.split('\n')
        current_q = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                current_q = line.split(':', 1)[1].strip()
            elif line.startswith('A') and ':' in line and current_q:
                answer = line.split(':', 1)[1].strip()
                qa_pairs.append({
                    'question': current_q,
                    'answer': answer,
                    'ev_keywords': self._extract_ev_keywords(current_q + ' ' + answer)
                })
                current_q = None
        
        return qa_pairs
    
    def _generate_fallback_qa(self, text: str, ev_keywords: List[str]) -> List[Dict[str, Any]]:
        """Generate simple Q-A pairs as fallback"""
        qa_pairs = []
        
        # Simple template-based Q-A generation
        if 'charging station' in text.lower():
            qa_pairs.append({
                'question': 'What are the key features of this charging station?',
                'answer': text[:200] + '...',
                'ev_keywords': ev_keywords
            })
        
        if 'level' in text.lower():
            qa_pairs.append({
                'question': 'What charging level is discussed in this text?',
                'answer': text[:200] + '...',
                'ev_keywords': ev_keywords
            })
        
        return qa_pairs
    
    def _extract_ev_keywords(self, text: str) -> List[str]:
        """Extract EV keywords from text"""
        ev_keywords = [
            'electric vehicle', 'EV', 'charging station', 'charger', 'battery',
            'plug-in', 'hybrid', 'fast charging', 'level 2', 'level 3',
            'DC fast charging', 'Tesla', 'Supercharger', 'ChargePoint',
            'EVgo', 'Electrify America', 'Blink', 'voltage', 'amperage',
            'kilowatt', 'kWh', 'range', 'miles per charge'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in ev_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _format_for_fine_tuning(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format Q-A pairs for OpenAI fine-tuning"""
        training_data = []
        
        for qa_pair in qa_pairs:
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert on electric vehicle charging infrastructure. Provide accurate, helpful information about EV charging stations, technology, policies, and related topics."
                    },
                    {
                        "role": "user",
                        "content": qa_pair['question']
                    },
                    {
                        "role": "assistant",
                        "content": qa_pair['answer']
                    }
                ]
            }
            
            training_data.append(training_example)
        
        return training_data
    
    def _generate_id(self, item: Dict[str, Any]) -> str:
        """Generate unique ID for item"""
        if 'url' in item:
            return hashlib.md5(item['url'].encode()).hexdigest()[:8]
        elif 'file_path' in item:
            return hashlib.md5(item['file_path'].encode()).hexdigest()[:8]
        else:
            return hashlib.md5(str(item).encode()).hexdigest()[:8]
    
    def save_processed_data(self, output_dir: str = "data/processed"):
        """Save processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        cleaned_file = output_path / "cleaned_data.json"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Save Q-A pairs
        qa_file = output_path / "qa_pairs.json"
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Save training data in JSONL format
        training_file = output_path / "training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for example in self._format_for_fine_tuning(self.qa_pairs):
                f.write(json.dumps(example) + '\n')
        
        self.logger.info(f"Processed data saved to {output_path}")
        
        return {
            'cleaned_data': str(cleaned_file),
            'qa_pairs': str(qa_file),
            'training_data': str(training_file)
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'total_input_items': len(self.cleaned_data) + len([d for d in self.cleaned_data if d.get('filtered_out')]),
            'cleaned_items': len(self.cleaned_data),
            'qa_pairs_generated': len(self.qa_pairs),
            'training_examples': len(self._format_for_fine_tuning(self.qa_pairs)),
            'avg_text_length': sum(len(d['cleaned_text']) for d in self.cleaned_data) / max(len(self.cleaned_data), 1),
            'unique_ev_keywords': len(set().union(*[d['ev_keywords'] for d in self.cleaned_data]))
        }
