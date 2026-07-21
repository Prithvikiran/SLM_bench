"""
Results Analysis and Report Generation
Analyzes experiment results and generates visualizations and reports
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes benchmarking results"""
    
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.results_df = None
        self.load_results()
    
    def load_results(self):
        """Load results from JSONL file"""
        results_list = []
        
        try:
            with open(self.results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        results_list.append(json.loads(line))
            
            # Flatten to dataframe
            flattened = self._flatten_results(results_list)
            self.results_df = pd.DataFrame(flattened)
            
            logger.info(f"Loaded {len(self.results_df)} experiment results")
        
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
    
    def _flatten_results(self, results: List[Dict]) -> List[Dict]:
        """Flatten nested result structure into dataframe rows"""
        flattened = []
        
        for result in results:
            if result['status'] != 'completed':
                continue
            
            base = result['config'].copy()
            metrics = result['metrics']
            
            # Extract task metrics
            task_metrics = metrics.get('task_metrics', {})
            base['f1'] = task_metrics.get('f1', np.nan)
            base['exact_match'] = task_metrics.get('exact_match', np.nan)
            
            # Extract latency stats
            latency_stats = metrics.get('latency_stats_ms', {})
            base['ttft_ms_mean'] = latency_stats.get('mean', np.nan)
            base['ttft_ms_median'] = latency_stats.get('median', np.nan)
            base['ttft_ms_p99'] = latency_stats.get('max', np.nan)
            
            # Extract throughput
            throughput_stats = metrics.get('throughput_stats_tps', {})
            base['tps_mean'] = throughput_stats.get('mean', np.nan)
            
            # Extract memory
            memory_stats = metrics.get('memory_stats_mb', {})
            base['peak_memory_mb'] = memory_stats.get('max', np.nan)
            
            flattened.append(base)
        
        return flattened
    
    def generate_summary_report(self, output_path: Path):
        """Generate summary report"""
        if self.results_df is None or len(self.results_df) == 0:
            logger.warning("No results to analyze")
            return
        
        report = []
        report.append("=" * 80)
        report.append("WINDOWS LLM INFERENCE BENCHMARKING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total experiments: {len(self.results_df)}")
        report.append(f"Models tested: {self.results_df['model_name'].nunique()}")
        report.append(f"Runtimes tested: {self.results_df['runtime'].nunique()}")
        report.append(f"Quantizations tested: {self.results_df['quantization'].nunique()}")
        report.append("")
        
        # Task performance summary
        report.append("TASK PERFORMANCE (SQuAD v2.0)")
        report.append("-" * 80)
        perf_df = self.results_df.groupby('model_name')[['f1', 'exact_match']].mean()
        report.append(perf_df.to_string())
        report.append("")
        
        # Runtime comparison
        report.append("RUNTIME PERFORMANCE (Average TTFT, Throughput)")
        report.append("-" * 80)
        runtime_df = self.results_df.groupby('runtime')[
            ['ttft_ms_mean', 'tps_mean']
        ].mean()
        report.append(runtime_df.to_string())
        report.append("")
        
        # Quantization impact
        report.append("QUANTIZATION IMPACT (Average F1, Memory)")
        report.append("-" * 80)
        quant_df = self.results_df.groupby('quantization')[
            ['f1', 'peak_memory_mb']
        ].mean()
        report.append(quant_df.to_string())
        report.append("")
        
        # Model-specific recommendations
        report.append("MODEL RECOMMENDATIONS")
        report.append("-" * 80)
        for model in self.results_df['model_name'].unique():
            model_results = self.results_df[self.results_df['model_name'] == model]
            best_f1_idx = model_results['f1'].idxmax()
            best_latency_idx = model_results['ttft_ms_mean'].idxmin()
            
            best_f1_row = model_results.loc[best_f1_idx]
            best_latency_row = model_results.loc[best_latency_idx]
            
            report.append(f"\n{model}:")
            report.append(f"  Best F1: {best_f1_row['f1']:.3f} "
                         f"({best_f1_row['runtime']}, {best_f1_row['quantization']})")
            report.append(f"  Fastest TTFT: {best_latency_row['ttft_ms_mean']:.1f}ms "
                         f"({best_latency_row['runtime']}, {best_latency_row['quantization']})")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = '\n'.join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {output_path}")
        print("\n" + report_text)
    
    def generate_decision_matrix(self, output_path: Path):
        """Generate deployment decision matrix"""
        if self.results_df is None:
            return
        
        matrix = []
        matrix.append("=" * 100)
        matrix.append("WINDOWS LLM DEPLOYMENT DECISION MATRIX")
        matrix.append("=" * 100)
        matrix.append("")
        
        # Scenario 1: Latency-sensitive (interactive chat)
        matrix.append("SCENARIO 1: Interactive Chat (Low Latency Critical)")
        matrix.append("-" * 100)
        latency_best = self.results_df.nsmallest(5, 'ttft_ms_mean')[
            ['model_name', 'runtime', 'quantization', 'ttft_ms_mean', 'f1', 'peak_memory_mb']
        ].drop_duplicates(subset=['model_name'])
        matrix.append(latency_best.to_string())
        matrix.append("")
        
        # Scenario 2: Throughput-sensitive (batch processing)
        matrix.append("SCENARIO 2: Batch Processing (Throughput Critical)")
        matrix.append("-" * 100)
        throughput_best = self.results_df.nlargest(5, 'tps_mean')[
            ['model_name', 'runtime', 'quantization', 'tps_mean', 'f1', 'peak_memory_mb']
        ].drop_duplicates(subset=['model_name'])
        matrix.append(throughput_best.to_string())
        matrix.append("")
        
        # Scenario 3: Quality-first (accurate answers)
        matrix.append("SCENARIO 3: Quality First (High Accuracy Required)")
        matrix.append("-" * 100)
        quality_best = self.results_df.nlargest(5, 'f1')[
            ['model_name', 'runtime', 'quantization', 'f1', 'ttft_ms_mean', 'peak_memory_mb']
        ].drop_duplicates(subset=['model_name'])
        matrix.append(quality_best.to_string())
        matrix.append("")
        
        # Scenario 4: Memory-constrained (small devices)
        matrix.append("SCENARIO 4: Memory Constrained (Small Devices)")
        matrix.append("-" * 100)
        memory_best = self.results_df.nsmallest(5, 'peak_memory_mb')[
            ['model_name', 'runtime', 'quantization', 'peak_memory_mb', 'f1', 'ttft_ms_mean']
        ].drop_duplicates(subset=['model_name'])
        matrix.append(memory_best.to_string())
        
        matrix.append("\n" + "=" * 100)
        
        # Save decision matrix
        matrix_text = '\n'.join(matrix)
        with open(output_path, 'w') as f:
            f.write(matrix_text)
        
        logger.info(f"Decision matrix saved to {output_path}")
        print("\n" + matrix_text)
    
    def generate_csv_export(self, output_path: Path):
        """Export results to CSV for external analysis"""
        if self.results_df is None:
            return
        
        # Select key columns
        export_cols = [
            'model_name', 'runtime', 'quantization', 'workload_type',
            'f1', 'exact_match',
            'ttft_ms_mean', 'ttft_ms_median', 'ttft_ms_p99',
            'tps_mean', 'peak_memory_mb'
        ]
        
        export_df = self.results_df[[col for col in export_cols if col in self.results_df.columns]]
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"CSV export saved to {output_path}")


def generate_all_reports(results_dir: Path):
    """Generate all analysis reports"""
    results_file = results_dir / "experiment_results.jsonl"
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    analyzer = ResultsAnalyzer(results_file)
    
    # Generate reports
    analyzer.generate_summary_report(results_dir / "summary_report.txt")
    analyzer.generate_decision_matrix(results_dir / "decision_matrix.txt")
    analyzer.generate_csv_export(results_dir / "results.csv")
    
    logger.info("All reports generated successfully")
