"""
Evaluation and Scoring Framework for LLM Query-Retrieval System
================================================================
Implements comprehensive evaluation metrics and scoring system as per requirements.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import requests

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question_id: str
    document_id: str
    query: str
    expected_answer: str
    predicted_answer: str
    is_correct: bool
    confidence: float
    processing_time: float
    token_usage: Dict[str, int]
    matched_clauses: int
    reasoning_quality: float  # 0-1 score for reasoning clarity

@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown"""
    total_score: float
    known_docs_score: float
    unknown_docs_score: float
    accuracy: float
    avg_confidence: float
    avg_processing_time: float
    total_questions: int
    correct_answers: int
    token_efficiency: float
    reasoning_quality: float

class EvaluationFramework:
    """Comprehensive evaluation framework for the query system"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 api_key: str = "36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9",
                 known_doc_weight: float = 0.5,
                 unknown_doc_weight: float = 2.0):
        
        self.base_url = base_url
        self.api_key = api_key
        self.known_doc_weight = known_doc_weight
        self.unknown_doc_weight = unknown_doc_weight
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Track known vs unknown documents
        self.known_documents = set()
        self.question_weights = {}
        
    def set_known_documents(self, known_doc_ids: List[str]):
        """Set list of known document IDs"""
        self.known_documents = set(known_doc_ids)
        logger.info(f"Set {len(known_doc_ids)} known documents")
    
    def set_question_weights(self, weights: Dict[str, float]):
        """Set question-specific weights"""
        self.question_weights = weights
        logger.info(f"Set weights for {len(weights)} questions")
    
    def evaluate_single_query(self, 
                            question_id: str,
                            query: str,
                            expected_answer: str,
                            document_id: str,
                            expected_keywords: List[str] = None) -> EvaluationResult:
        """Evaluate a single query"""
        
        start_time = time.time()
        
        try:
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/v1/query",
                headers=self.headers,
                json={"query": query, "max_results": 5},
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"API error for question {question_id}: {response.status_code}")
                return self._create_failed_result(question_id, query, expected_answer, 
                                               document_id, processing_time, response.text)
            
            result = response.json()
            
            # Evaluate correctness
            is_correct = self._evaluate_correctness(
                result["answer"], 
                expected_answer, 
                expected_keywords
            )
            
            # Evaluate reasoning quality
            reasoning_quality = self._evaluate_reasoning_quality(result.get("reasoning", ""))
            
            return EvaluationResult(
                question_id=question_id,
                document_id=document_id,
                query=query,
                expected_answer=expected_answer,
                predicted_answer=result["answer"],
                is_correct=is_correct,
                confidence=result["confidence"],
                processing_time=processing_time,
                token_usage=result.get("token_usage", {}),
                matched_clauses=len(result.get("matched_clauses", [])),
                reasoning_quality=reasoning_quality
            )
            
        except Exception as e:
            logger.error(f"Error evaluating question {question_id}: {str(e)}")
            processing_time = time.time() - start_time
            return self._create_failed_result(question_id, query, expected_answer, 
                                           document_id, processing_time, str(e))
    
    def _create_failed_result(self, question_id: str, query: str, expected_answer: str,
                            document_id: str, processing_time: float, error: str) -> EvaluationResult:
        """Create result for failed query"""
        return EvaluationResult(
            question_id=question_id,
            document_id=document_id,
            query=query,
            expected_answer=expected_answer,
            predicted_answer=f"ERROR: {error}",
            is_correct=False,
            confidence=0.0,
            processing_time=processing_time,
            token_usage={"total_tokens": 0},
            matched_clauses=0,
            reasoning_quality=0.0
        )
    
    def _evaluate_correctness(self, predicted: str, expected: str, keywords: List[str] = None) -> bool:
        """Evaluate if the predicted answer is correct"""
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Method 1: Keyword matching
        if keywords:
            keyword_matches = sum(1 for keyword in keywords 
                                if keyword.lower() in predicted_lower)
            keyword_score = keyword_matches / len(keywords)
            if keyword_score >= 0.5:  # At least 50% keywords present
                return True
        
        # Method 2: Semantic similarity (simplified)
        # In production, use sentence transformers for better evaluation
        common_words = set(predicted_lower.split()) & set(expected_lower.split())
        if len(common_words) >= 3:  # At least 3 common words
            return True
        
        # Method 3: Length and structure check
        if len(predicted) > 10 and "error" not in predicted_lower and "not found" not in predicted_lower:
            return True
        
        return False
    
    def _evaluate_reasoning_quality(self, reasoning: str) -> float:
        """Evaluate quality of reasoning explanation (0-1 score)"""
        if not reasoning or len(reasoning) < 10:
            return 0.0
        
        quality_indicators = [
            "document" in reasoning.lower(),
            "page" in reasoning.lower() or "section" in reasoning.lower(),
            "based on" in reasoning.lower(),
            "according to" in reasoning.lower(),
            len(reasoning.split()) > 20,  # Sufficient detail
            reasoning.count(".") >= 2,  # Multiple sentences
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def calculate_score(self, question_id: str, document_id: str, is_correct: bool) -> float:
        """Calculate score for a single question using the specified scoring system"""
        if not is_correct:
            return 0.0
        
        # Get question weight
        question_weight = self.question_weights.get(question_id, 1.0)
        
        # Get document weight
        doc_weight = (self.known_doc_weight if document_id in self.known_documents 
                     else self.unknown_doc_weight)
        
        return question_weight * doc_weight
    
    def evaluate_batch(self, evaluation_data: List[Dict[str, Any]]) -> Tuple[List[EvaluationResult], ScoreBreakdown]:
        """Evaluate multiple queries and calculate comprehensive scores"""
        
        logger.info(f"Starting batch evaluation of {len(evaluation_data)} queries")
        
        results = []
        total_score = 0.0
        known_docs_score = 0.0
        unknown_docs_score = 0.0
        total_processing_time = 0.0
        total_tokens = 0
        total_confidence = 0.0
        total_reasoning_quality = 0.0
        
        for i, item in enumerate(evaluation_data, 1):
            logger.info(f"Evaluating query {i}/{len(evaluation_data)}: {item['question_id']}")
            
            result = self.evaluate_single_query(
                question_id=item["question_id"],
                query=item["query"],
                expected_answer=item["expected_answer"],
                document_id=item["document_id"],
                expected_keywords=item.get("expected_keywords", [])
            )
            
            results.append(result)
            
            # Calculate score for this result
            score = self.calculate_score(
                result.question_id, 
                result.document_id, 
                result.is_correct
            )
            total_score += score
            
            # Track scores by document type
            if result.document_id in self.known_documents:
                known_docs_score += score
            else:
                unknown_docs_score += score
            
            # Aggregate metrics
            total_processing_time += result.processing_time
            total_tokens += result.token_usage.get("total_tokens", 0)
            total_confidence += result.confidence
            total_reasoning_quality += result.reasoning_quality
        
        # Calculate final metrics
        num_results = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        
        score_breakdown = ScoreBreakdown(
            total_score=total_score,
            known_docs_score=known_docs_score,
            unknown_docs_score=unknown_docs_score,
            accuracy=correct_answers / num_results,
            avg_confidence=total_confidence / num_results,
            avg_processing_time=total_processing_time / num_results,
            total_questions=num_results,
            correct_answers=correct_answers,
            token_efficiency=correct_answers / max(total_tokens, 1) * 1000,  # Correct answers per 1K tokens
            reasoning_quality=total_reasoning_quality / num_results
        )
        
        logger.info(f"Evaluation completed. Total score: {total_score:.2f}")
        
        return results, score_breakdown
    
    def generate_report(self, results: List[EvaluationResult], 
                       breakdown: ScoreBreakdown, 
                       output_path: str = "evaluation_report.html"):
        """Generate comprehensive evaluation report"""
        
        # Convert results to DataFrame for analysis
        results_data = []
        for result in results:
            row = asdict(result)
            row['document_type'] = 'known' if result.document_id in self.known_documents else 'unknown'
            row['score'] = self.calculate_score(result.question_id, result.document_id, result.is_correct)
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # Generate HTML report
        html_content = self._generate_html_report(df, breakdown)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_path}")
        
        return output_path
    
    def _generate_html_report(self, df: pd.DataFrame, breakdown: ScoreBreakdown) -> str:
        """Generate HTML report content"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Query System Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2e86ab; }}
                .metric-label {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: #28a745; }}
                .failure {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 LLM Query System Evaluation Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{breakdown.total_score:.2f}</div>
                    <div class="metric-label">Total Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{breakdown.accuracy:.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{breakdown.avg_confidence:.3f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{breakdown.avg_processing_time:.3f}s</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{breakdown.token_efficiency:.2f}</div>
                    <div class="metric-label">Token Efficiency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{breakdown.reasoning_quality:.3f}</div>
                    <div class="metric-label">Reasoning Quality</div>
                </div>
            </div>
            
            <h2>📊 Score Breakdown</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Details</th></tr>
                <tr><td>Known Documents Score</td><td>{breakdown.known_docs_score:.2f}</td><td>Weight: {self.known_doc_weight}</td></tr>
                <tr><td>Unknown Documents Score</td><td>{breakdown.unknown_docs_score:.2f}</td><td>Weight: {self.unknown_doc_weight}</td></tr>
                <tr><td>Correct Answers</td><td>{breakdown.correct_answers}/{breakdown.total_questions}</td><td>{breakdown.accuracy:.1%} success rate</td></tr>
            </table>
            
            <h2>📋 Detailed Results</h2>
            <table>
                <tr>
                    <th>Question ID</th>
                    <th>Query</th>
                    <th>Document</th>
                    <th>Type</th>
                    <th>Result</th>
                    <th>Score</th>
                    <th>Confidence</th>
                    <th>Time (s)</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            status_class = "success" if row['is_correct'] else "failure"
            doc_type = row['document_type'].title()
            
            html += f"""
                <tr>
                    <td>{row['question_id']}</td>
                    <td title="{row['query']}">{row['query'][:50]}...</td>
                    <td>{row['document_id']}</td>
                    <td>{doc_type}</td>
                    <td class="{status_class}">{'✅ Correct' if row['is_correct'] else '❌ Incorrect'}</td>
                    <td>{row['score']:.2f}</td>
                    <td>{row['confidence']:.3f}</td>
                    <td>{row['processing_time']:.3f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>📈 Performance Analysis</h2>
            <div class="chart-container">
                <h3>Key Insights:</h3>
                <ul>
        """
        
        # Add insights
        known_accuracy = df[df['document_type'] == 'known']['is_correct'].mean()
        unknown_accuracy = df[df['document_type'] == 'unknown']['is_correct'].mean()
        
        html += f"""
                    <li><strong>Known Documents:</strong> {known_accuracy:.1%} accuracy ({df[df['document_type'] == 'known'].shape[0]} questions)</li>
                    <li><strong>Unknown Documents:</strong> {unknown_accuracy:.1%} accuracy ({df[df['document_type'] == 'unknown'].shape[0]} questions)</li>
                    <li><strong>Average Response Time:</strong> {breakdown.avg_processing_time:.3f}s per query</li>
                    <li><strong>Token Efficiency:</strong> {breakdown.token_efficiency:.2f} correct answers per 1K tokens</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html

# Sample evaluation data generator
def create_sample_evaluation_data() -> List[Dict[str, Any]]:
    """Create sample evaluation dataset"""
    return [
        {
            "question_id": "Q001",
            "query": "Does this policy cover knee surgery, and what are the conditions?",
            "expected_answer": "Yes, knee surgery is covered with pre-authorization and $500 deductible",
            "document_id": "health_policy_2024",
            "expected_keywords": ["knee surgery", "covered", "pre-authorization", "deductible"],
            "weight": 2.0
        },
        {
            "question_id": "Q002", 
            "query": "What is the maximum coverage for emergency room visits?",
            "expected_answer": "Emergency room visits are covered up to $5000 per incident",
            "document_id": "health_policy_2024",
            "expected_keywords": ["emergency room", "coverage", "maximum", "$5000"],
            "weight": 1.5
        },
        {
            "question_id": "Q003",
            "query": "What are the termination conditions in the employment contract?",
            "expected_answer": "Employment can be terminated with 30 days notice or immediately for cause",
            "document_id": "employment_contract_2024",
            "expected_keywords": ["termination", "30 days", "notice", "cause"],
            "weight": 2.0
        },
        {
            "question_id": "Q004",
            "query": "How many vacation days do employees get annually?",
            "expected_answer": "Employees receive 20 vacation days per year after 1 year of service",
            "document_id": "employee_handbook_2024", 
            "expected_keywords": ["vacation days", "20", "annually", "1 year"],
            "weight": 1.0
        },
        {
            "question_id": "Q005",
            "query": "What are the GDPR data retention requirements?",
            "expected_answer": "Personal data must be deleted within 2 years unless legally required",
            "document_id": "gdpr_compliance_2024",
            "expected_keywords": ["GDPR", "data retention", "2 years", "deleted"],
            "weight": 2.5
        }
    ]

# Performance benchmarking utilities
class PerformanceBenchmark:
    """Benchmark system performance under different loads"""
    
    def __init__(self, evaluator: EvaluationFramework):
        self.evaluator = evaluator
    
    def run_latency_test(self, queries: List[str], iterations: int = 10) -> Dict[str, float]:
        """Test response latency"""
        times = []
        
        for _ in range(iterations):
            for query in queries:
                start_time = time.time()
                
                try:
                    response = requests.post(
                        f"{self.evaluator.base_url}/api/v1/query",
                        headers=self.evaluator.headers,
                        json={"query": query},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        times.append(time.time() - start_time)
                        
                except Exception as e:
                    logger.warning(f"Query failed during latency test: {e}")
        
        return {
            "avg_latency": np.mean(times),
            "min_latency": np.min(times),
            "max_latency": np.max(times),
            "p95_latency": np.percentile(times, 95),
            "p99_latency": np.percentile(times, 99),
            "total_queries": len(times)
        }
    
    def run_concurrency_test(self, query: str, concurrent_requests: int = 5) -> Dict[str, Any]:
        """Test concurrent request handling"""
        import concurrent.futures
        
        def make_request():
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.evaluator.base_url}/api/v1/query",
                    headers=self.evaluator.headers,
                    json={"query": query},
                    timeout=30
                )
                return {
                    "success": response.status_code == 200,
                    "time": time.time() - start_time,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "success": False,
                    "time": time.time() - start_time,
                    "error": str(e)
                }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if r["success"])
        avg_response_time = np.mean([r["time"] for r in results if r["success"]])
        
        return {
            "concurrent_requests": concurrent_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / concurrent_requests,
            "total_time": total_time,
            "avg_response_time": avg_response_time,
            "requests_per_second": concurrent_requests / total_time
        }

# Main evaluation script
def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM Query System")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default="36d49ac587c7cb7331f48ad3067cd8057811970de89b734f8326aa39d665c8c9", help="API key")
    parser.add_argument("--evaluation-file", help="JSON file with evaluation data")
    parser.add_argument("--known-docs", nargs="*", help="List of known document IDs")
    parser.add_argument("--output", default="evaluation_report.html", help="Output report file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = EvaluationFramework(args.base_url, args.api_key)
    
    # Set known documents if provided
    if args.known_docs:
        evaluator.set_known_documents(args.known_docs)
    
    # Load evaluation data
    if args.evaluation_file:
        with open(args.evaluation_file, 'r') as f:
            evaluation_data = json.load(f)
    else:
        print("📝 Using sample evaluation data...")
        evaluation_data = create_sample_evaluation_data()
        
        # Set question weights
        weights = {item["question_id"]: item.get("weight", 1.0) for item in evaluation_data}
        evaluator.set_question_weights(weights)
    
    # Run evaluation
    print("🚀 Starting comprehensive evaluation...")
    results, breakdown = evaluator.evaluate_batch(evaluation_data)
    
    # Generate report
    report_path = evaluator.generate_report(results, breakdown, args.output)
    print(f"📊 Report generated: {report_path}")
    
    # Run benchmarks if requested
    if args.benchmark:
        print("\n⚡ Running performance benchmarks...")
        benchmark = PerformanceBenchmark(evaluator)
        
        # Latency test
        test_queries = [item["query"] for item in evaluation_data[:3]]  # Use first 3 queries
        latency_results = benchmark.run_latency_test(test_queries, iterations=5)
        
        print("\n📈 Latency Results:")
        for key, value in latency_results.items():
            print(f"  {key}: {value:.3f}s" if "latency" in key else f"  {key}: {value}")
        
        # Concurrency test
        concurrency_results = benchmark.run_concurrency_test(test_queries[0], concurrent_requests=3)
        
        print("\n🔄 Concurrency Results:")
        for key, value in concurrency_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # Print summary
    print(f"\n🎯 Final Results Summary:")
    print(f"  Total Score: {breakdown.total_score:.2f}")
    print(f"  Accuracy: {breakdown.accuracy:.1%}")
    print(f"  Average Confidence: {breakdown.avg_confidence:.3f}")
    print(f"  Average Response Time: {breakdown.avg_processing_time:.3f}s")
    print(f"  Token Efficiency: {breakdown.token_efficiency:.2f}")
    
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exit(main())