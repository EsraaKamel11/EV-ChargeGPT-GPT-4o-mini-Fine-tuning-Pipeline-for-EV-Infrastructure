"""
Model Evaluator for AI Pipeline
Handles domain benchmarking, evaluation metrics, and LLM-as-judge evaluation
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import openai
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import asyncio
import aiohttp

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    openai_model: str = "gpt-4o-mini"
    baseline_model: str = "gpt-4o-mini"
    fine_tuned_model: Optional[str] = None
    evaluation_prompts: List[str] = None
    metrics: List[str] = None
    llm_judge_model: str = "gpt-4o-mini"
    max_evaluation_samples: int = 50
    temperature: float = 0.7
    max_tokens: int = 500

class ModelEvaluator:
    def __init__(self, openai_api_key: str, config: EvaluationConfig):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Evaluation results
        self.evaluation_results = {}
        self.benchmark_data = []
        
        # Initialize default evaluation prompts if none provided
        if not self.config.evaluation_prompts:
            self.config.evaluation_prompts = self._get_default_ev_prompts()
        
        # Initialize default metrics if none provided
        if not self.config.metrics:
            self.config.metrics = ['rouge', 'bleu', 'llm_judge', 'latency']
    
    def _get_default_ev_prompts(self) -> List[str]:
        """Get default EV charging evaluation prompts"""
        return [
            "What are the different levels of EV charging and their power outputs?",
            "How do I install a Level 2 charging station at home?",
            "What safety considerations are important for EV charging?",
            "What are the main connector types for EV charging?",
            "How does DC fast charging work and what are its benefits?",
            "What are the costs associated with installing EV charging infrastructure?",
            "How do I find public charging stations in my area?",
            "What maintenance is required for EV charging equipment?",
            "What are the environmental benefits of electric vehicles?",
            "How do I calculate charging time for my EV?"
        ]
    
    def create_domain_benchmark(self, output_file: str = "data/benchmark/ev_charging_benchmark.json") -> Dict[str, Any]:
        """Create a domain-specific benchmark dataset for EV charging"""
        self.logger.info("Creating EV charging domain benchmark")
        
        benchmark_data = []
        
        # Generate benchmark Q-A pairs using GPT-4o-mini
        for prompt in tqdm(self.config.evaluation_prompts, desc="Generating benchmark"):
            try:
                # Generate high-quality answer using baseline model
                response = self.openai_client.chat.completions.create(
                    model=self.config.baseline_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert on electric vehicle charging infrastructure. Provide comprehensive, accurate, and detailed answers."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=0.3  # Lower temperature for more consistent answers
                )
                
                answer = response.choices[0].message.content
                
                benchmark_item = {
                    'id': f"benchmark_{len(benchmark_data)}",
                    'question': prompt,
                    'reference_answer': answer,
                    'category': self._categorize_question(prompt),
                    'difficulty': self._assess_difficulty(prompt),
                    'keywords': self._extract_keywords(prompt + ' ' + answer)
                }
                
                benchmark_data.append(benchmark_item)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error generating benchmark for prompt '{prompt}': {e}")
                continue
        
        self.benchmark_data = benchmark_data
        
        # Save benchmark
        benchmark_path = Path(output_file)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Benchmark created with {len(benchmark_data)} items: {benchmark_path}")
        
        return {
            'success': True,
            'benchmark_items': len(benchmark_data),
            'output_file': str(benchmark_path)
        }
    
    def _categorize_question(self, question: str) -> str:
        """Categorize question by topic"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['level', 'power', 'output', 'voltage']):
            return "technical_specs"
        elif any(word in question_lower for word in ['install', 'setup', 'equipment']):
            return "installation"
        elif any(word in question_lower for word in ['safety', 'maintenance', 'care']):
            return "safety_maintenance"
        elif any(word in question_lower for word in ['connector', 'plug', 'type']):
            return "connectors"
        elif any(word in question_lower for word in ['cost', 'price', 'financial']):
            return "costs"
        elif any(word in question_lower for word in ['find', 'location', 'public']):
            return "location_finding"
        elif any(word in question_lower for word in ['environmental', 'benefit', 'impact']):
            return "environmental"
        else:
            return "general"
    
    def _assess_difficulty(self, question: str) -> str:
        """Assess question difficulty"""
        question_lower = question.lower()
        
        # Simple heuristics for difficulty assessment
        technical_terms = ['voltage', 'amperage', 'kilowatt', 'infrastructure', 'protocols']
        basic_terms = ['what', 'how', 'where', 'when']
        
        if any(term in question_lower for term in technical_terms):
            return "advanced"
        elif any(term in question_lower for term in basic_terms):
            return "basic"
        else:
            return "intermediate"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract EV-related keywords from text"""
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
    
    def evaluate_models(self, benchmark_file: str = None) -> Dict[str, Any]:
        """Evaluate baseline vs fine-tuned models"""
        if not benchmark_file:
            if not self.benchmark_data:
                self.logger.error("No benchmark data available. Please create benchmark first.")
                return {'success': False, 'error': 'No benchmark data'}
            benchmark_data = self.benchmark_data
        else:
            # Load benchmark from file
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading benchmark file: {e}")
                return {'success': False, 'error': f'Error loading benchmark: {e}'}
        
        self.logger.info(f"Starting model evaluation with {len(benchmark_data)} benchmark items")
        
        # Limit evaluation samples
        evaluation_samples = benchmark_data[:self.config.max_evaluation_samples]
        
        # Evaluate baseline model
        baseline_results = self._evaluate_model_responses(
            self.config.baseline_model, 
            evaluation_samples, 
            "baseline"
        )
        
        # Evaluate fine-tuned model if available
        fine_tuned_results = None
        if self.config.fine_tuned_model:
            fine_tuned_results = self._evaluate_model_responses(
                self.config.fine_tuned_model,
                evaluation_samples,
                "fine_tuned"
            )
        
        # Compile evaluation results
        evaluation_summary = {
            'success': True,
            'benchmark_items': len(evaluation_samples),
            'baseline_model': self.config.baseline_model,
            'fine_tuned_model': self.config.fine_tuned_model,
            'baseline_results': baseline_results,
            'fine_tuned_results': fine_tuned_results,
            'comparison': self._compare_models(baseline_results, fine_tuned_results)
        }
        
        self.evaluation_results = evaluation_summary
        
        return evaluation_summary
    
    def _evaluate_model_responses(self, model_name: str, benchmark_data: List[Dict], model_type: str) -> Dict[str, Any]:
        """Evaluate responses from a specific model"""
        self.logger.info(f"Evaluating {model_type} model: {model_name}")
        
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'responses': [],
            'metrics': {},
            'latency': []
        }
        
        for item in tqdm(benchmark_data, desc=f"Evaluating {model_type}"):
            try:
                # Measure response time
                start_time = time.time()
                
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert on electric vehicle charging infrastructure. Provide accurate, helpful information."
                        },
                        {
                            "role": "user",
                            "content": item['question']
                        }
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                answer = response.choices[0].message.content
                
                # Store response
                response_data = {
                    'question_id': item['id'],
                    'question': item['question'],
                    'reference_answer': item['reference_answer'],
                    'model_answer': answer,
                    'latency': latency
                }
                
                results['responses'].append(response_data)
                results['latency'].append(latency)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {model_type} for question {item['id']}: {e}")
                continue
        
        # Calculate metrics
        if results['responses']:
            results['metrics'] = self._calculate_metrics(results['responses'])
            results['metrics']['avg_latency'] = np.mean(results['latency'])
            results['metrics']['total_responses'] = len(results['responses'])
        
        return results
    
    def _calculate_metrics(self, responses: List[Dict]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Calculate ROUGE scores (simplified)
        rouge_scores = []
        for response in responses:
            rouge_score = self._calculate_rouge_simple(
                response['reference_answer'], 
                response['model_answer']
            )
            rouge_scores.append(rouge_score)
        
        if rouge_scores:
            metrics['rouge_avg'] = np.mean(rouge_scores)
            metrics['rouge_std'] = np.std(rouge_scores)
        
        # Calculate BLEU score (simplified)
        bleu_scores = []
        for response in responses:
            bleu_score = self._calculate_bleu_simple(
                response['reference_answer'], 
                response['model_answer']
            )
            bleu_scores.append(bleu_score)
        
        if bleu_scores:
            metrics['bleu_avg'] = np.mean(bleu_scores)
            metrics['bleu_std'] = np.std(bleu_scores)
        
        # LLM-as-judge evaluation
        llm_judge_scores = []
        for response in responses[:10]:  # Limit to first 10 for cost efficiency
            judge_score = self._evaluate_with_llm_judge(
                response['question'],
                response['reference_answer'],
                response['model_answer']
            )
            llm_judge_scores.append(judge_score)
        
        if llm_judge_scores:
            metrics['llm_judge_avg'] = np.mean(llm_judge_scores)
            metrics['llm_judge_std'] = np.std(llm_judge_scores)
        
        return metrics
    
    def _calculate_rouge_simple(self, reference: str, candidate: str) -> float:
        """Calculate simplified ROUGE score"""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(ref_words.intersection(cand_words))
        return overlap / len(ref_words)
    
    def _calculate_bleu_simple(self, reference: str, candidate: str) -> float:
        """Calculate simplified BLEU score"""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not cand_words:
            return 0.0
        
        # Simple n-gram overlap
        matches = 0
        for i in range(len(cand_words)):
            if i < len(ref_words) and cand_words[i] == ref_words[i]:
                matches += 1
        
        return matches / len(cand_words)
    
    def _evaluate_with_llm_judge(self, question: str, reference: str, candidate: str) -> float:
        """Evaluate response quality using LLM-as-judge"""
        try:
            prompt = f"""
Evaluate the quality of this answer to the question about electric vehicle charging.

Question: {question}

Reference Answer: {reference}

Candidate Answer: {candidate}

Rate the candidate answer on a scale of 1-10, where:
1 = Poor: Inaccurate, unhelpful, or irrelevant
5 = Average: Somewhat accurate but incomplete
10 = Excellent: Accurate, comprehensive, and helpful

Provide only the numerical score (1-10):
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.config.llm_judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide only numerical scores."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            
            # Extract numerical score
            try:
                score = float(score_text)
                return max(1.0, min(10.0, score))  # Clamp between 1-10
            except ValueError:
                return 5.0  # Default score if parsing fails
                
        except Exception as e:
            self.logger.warning(f"Error in LLM-as-judge evaluation: {e}")
            return 5.0  # Default score
    
    def _compare_models(self, baseline_results: Dict, fine_tuned_results: Dict) -> Dict[str, Any]:
        """Compare baseline and fine-tuned model performance"""
        if not fine_tuned_results:
            return {
                'comparison_available': False,
                'message': 'No fine-tuned model results to compare'
            }
        
        comparison = {
            'comparison_available': True,
            'improvements': {},
            'regressions': {},
            'summary': {}
        }
        
        baseline_metrics = baseline_results.get('metrics', {})
        fine_tuned_metrics = fine_tuned_results.get('metrics', {})
        
        # Compare key metrics
        for metric in ['rouge_avg', 'bleu_avg', 'llm_judge_avg']:
            if metric in baseline_metrics and metric in fine_tuned_metrics:
                baseline_val = baseline_metrics[metric]
                fine_tuned_val = fine_tuned_metrics[metric]
                
                improvement = fine_tuned_val - baseline_val
                improvement_pct = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
                
                if improvement > 0:
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'fine_tuned': fine_tuned_val,
                        'improvement': improvement,
                        'improvement_pct': improvement_pct
                    }
                else:
                    comparison['regressions'][metric] = {
                        'baseline': baseline_val,
                        'fine_tuned': fine_tuned_val,
                        'regression': abs(improvement),
                        'regression_pct': abs(improvement_pct)
                    }
        
        # Latency comparison
        if 'avg_latency' in baseline_metrics and 'avg_latency' in fine_tuned_metrics:
            baseline_latency = baseline_metrics['avg_latency']
            fine_tuned_latency = fine_tuned_metrics['avg_latency']
            
            latency_diff = fine_tuned_latency - baseline_latency
            if latency_diff > 0:
                comparison['regressions']['latency'] = {
                    'baseline': baseline_latency,
                    'fine_tuned': fine_tuned_latency,
                    'slowdown': latency_diff
                }
            else:
                comparison['improvements']['latency'] = {
                    'baseline': baseline_latency,
                    'fine_tuned': fine_tuned_latency,
                    'speedup': abs(latency_diff)
                }
        
        # Summary
        total_improvements = len(comparison['improvements'])
        total_regressions = len(comparison['regressions'])
        
        comparison['summary'] = {
            'total_metrics_improved': total_improvements,
            'total_metrics_regressed': total_regressions,
            'overall_trend': 'improvement' if total_improvements > total_regressions else 'regression'
        }
        
        return comparison
    
    def save_evaluation_results(self, output_file: str = "data/evaluation/evaluation_results.json") -> bool:
        """Save evaluation results to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Evaluation results saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")
            return False
    
    def generate_evaluation_report(self, output_file: str = "data/evaluation/evaluation_report.md") -> str:
        """Generate a markdown evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report_lines = [
            "# Model Evaluation Report",
            "",
            f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Baseline Model:** {self.config.baseline_model}",
            f"**Fine-tuned Model:** {self.config.fine_tuned_model or 'N/A'}",
            "",
            "## Benchmark Summary",
            f"- **Total Benchmark Items:** {self.evaluation_results.get('benchmark_items', 0)}",
            f"- **Evaluation Samples:** {min(self.config.max_evaluation_samples, self.evaluation_results.get('benchmark_items', 0))}",
            "",
            "## Baseline Model Results",
        ]
        
        baseline_results = self.evaluation_results.get('baseline_results', {})
        if baseline_results and 'metrics' in baseline_results:
            metrics = baseline_results['metrics']
            report_lines.extend([
                f"- **ROUGE Score:** {metrics.get('rouge_avg', 'N/A'):.3f} ± {metrics.get('rouge_std', 'N/A'):.3f}",
                f"- **BLEU Score:** {metrics.get('bleu_avg', 'N/A'):.3f} ± {metrics.get('bleu_std', 'N/A'):.3f}",
                f"- **LLM Judge Score:** {metrics.get('llm_judge_avg', 'N/A'):.3f} ± {metrics.get('llm_judge_std', 'N/A'):.3f}",
                f"- **Average Latency:** {metrics.get('avg_latency', 'N/A'):.3f}s",
                f"- **Total Responses:** {metrics.get('total_responses', 'N/A')}",
                ""
            ])
        
        # Add fine-tuned results if available
        fine_tuned_results = self.evaluation_results.get('fine_tuned_results')
        if fine_tuned_results:
            report_lines.extend([
                "## Fine-tuned Model Results",
            ])
            
            if 'metrics' in fine_tuned_results:
                metrics = fine_tuned_results['metrics']
                report_lines.extend([
                    f"- **ROUGE Score:** {metrics.get('rouge_avg', 'N/A'):.3f} ± {metrics.get('rouge_std', 'N/A'):.3f}",
                    f"- **BLEU Score:** {metrics.get('bleu_avg', 'N/A'):.3f} ± {metrics.get('bleu_std', 'N/A'):.3f}",
                    f"- **LLM Judge Score:** {metrics.get('llm_judge_avg', 'N/A'):.3f} ± {metrics.get('llm_judge_std', 'N/A'):.3f}",
                    f"- **Average Latency:** {metrics.get('avg_latency', 'N/A'):.3f}s",
                    f"- **Total Responses:** {metrics.get('total_responses', 'N/A')}",
                    ""
                ])
        
        # Add comparison results
        comparison = self.evaluation_results.get('comparison', {})
        if comparison.get('comparison_available'):
            report_lines.extend([
                "## Model Comparison",
                "",
                f"**Overall Trend:** {comparison['summary'].get('overall_trend', 'N/A').title()}",
                f"**Metrics Improved:** {comparison['summary'].get('total_metrics_improved', 0)}",
                f"**Metrics Regressed:** {comparison['summary'].get('total_metrics_regressed', 0)}",
                ""
            ])
            
            if comparison.get('improvements'):
                report_lines.append("### Improvements")
                for metric, details in comparison['improvements'].items():
                    if 'improvement_pct' in details:
                        report_lines.append(f"- **{metric}:** +{details['improvement_pct']:.1f}%")
                    elif 'speedup' in details:
                        report_lines.append(f"- **{metric}:** {details['speedup']:.3f}s faster")
                report_lines.append("")
            
            if comparison.get('regressions'):
                report_lines.append("### Regressions")
                for metric, details in comparison['regressions'].items():
                    if 'regression_pct' in details:
                        report_lines.append(f"- **{metric}:** -{details['regression_pct']:.1f}%")
                    elif 'slowdown' in details:
                        report_lines.append(f"- **{metric}:** {details['slowdown']:.3f}s slower")
                report_lines.append("")
        
        # Save report
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Evaluation report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation report: {e}")
        
        return '\n'.join(report_lines)
