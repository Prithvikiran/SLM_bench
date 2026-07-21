"""
System Metrics Collector
Collects latency, throughput, memory, and GPU metrics on Windows
"""

import psutil
import time
import threading
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """Collects system metrics during inference"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.monitoring = False
        self.monitor_thread = None
        
        # Timestamps
        self.request_start_time = None
        self.first_token_time = None
        self.inference_end_time = None
        self.model_load_start = None
        self.model_load_end = None
        
        # Token counts
        self.input_tokens = 0
        self.output_tokens = 0
    
    def start_request_timer(self):
        """Start timing from request submission"""
        self.request_start_time = time.perf_counter()
    
    def record_first_token(self):
        """Record time of first token"""
        self.first_token_time = time.perf_counter()
    
    def end_request_timer(self):
        """End timing after last token"""
        self.inference_end_time = time.perf_counter()
    
    def start_monitoring(self, interval: float = 0.1):
        """
        Start background thread to monitor system metrics
        
        Args:
            interval: Time between samples (seconds)
        """
        if self.monitoring:
            logger.warning("Already monitoring")
            return
        
        self.monitoring = True
        self.metrics_history.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        try:
            process = psutil.Process()
            
            while self.monitoring:
                sample = {
                    'timestamp': time.perf_counter(),
                    'cpu_percent': process.cpu_percent(interval=None),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                }
                
                # Try GPU metrics if available
                try:
                    sample.update(self._get_gpu_metrics())
                except:
                    pass
                
                self.metrics_history.append(sample)
                time.sleep(interval)
        
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def _get_gpu_metrics(self) -> Dict:
        """
        Collect GPU metrics (NVIDIA CUDA)
        Returns empty dict if GPU not available
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            
            return {
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_utilization': self._get_gpu_utilization()
            }
        except:
            return {}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage (via nvidia-smi if available)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split(',')[0])
        except:
            pass
        return 0.0
    
    def get_latency_metrics(self) -> Dict[str, float]:
        """Get latency metrics"""
        if not self.request_start_time:
            return {}
        
        ttft = (self.first_token_time - self.request_start_time) * 1000  # ms
        total_latency = (self.inference_end_time - self.request_start_time) * 1000  # ms
        decode_latency = (self.inference_end_time - self.first_token_time) * 1000  # ms
        
        return {
            'ttft_ms': round(ttft, 2),
            'total_latency_ms': round(total_latency, 2),
            'decode_latency_ms': round(decode_latency, 2)
        }
    
    def get_throughput_metrics(self) -> Dict[str, float]:
        """Get throughput metrics"""
        if not self.request_start_time or not self.inference_end_time:
            return {}

        # For non-streaming runtimes, first_token_time == inference_end_time,
        # so decode_time ≈ 0. Use total request time instead.
        total_time = self.inference_end_time - self.request_start_time

        if total_time > 0:
            tokens_per_second = self.output_tokens / total_time
        else:
            tokens_per_second = 0
        
        return {
            'tokens_per_second': round(tokens_per_second, 2),
            'end_to_end_tps': round(tokens_per_second, 2),
            'sentences_per_second': round(tokens_per_second / 15, 2)  # Rough estimate
        }
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics"""
        if not self.metrics_history:
            return {}
        
        memory_values = [m['memory_mb'] for m in self.metrics_history]
        
        metrics = {
            'peak_memory_mb': round(max(memory_values), 2),
            'avg_memory_mb': round(sum(memory_values) / len(memory_values), 2),
            'min_memory_mb': round(min(memory_values), 2)
        }
        
        # GPU memory if available
        if 'gpu_memory_mb' in self.metrics_history[0]:
            gpu_memory = [m.get('gpu_memory_mb', 0) for m in self.metrics_history]
            metrics['peak_gpu_memory_mb'] = round(max(gpu_memory), 2)
        
        return metrics
    
    def get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU utilization metrics"""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics_history]
        
        return {
            'peak_cpu_percent': round(max(cpu_values), 2),
            'avg_cpu_percent': round(sum(cpu_values) / len(cpu_values), 2),
            'min_cpu_percent': round(min(cpu_values), 2)
        }
    
    def get_all_metrics(self) -> Dict:
        """Get all collected metrics"""
        return {
            'latency': self.get_latency_metrics(),
            'throughput': self.get_throughput_metrics(),
            'memory': self.get_memory_metrics(),
            'cpu': self.get_cpu_metrics()
        }
    
    def reset(self):
        """Reset all metrics"""
        self.request_start_time = None
        self.first_token_time = None
        self.inference_end_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.metrics_history.clear()


class ColdStartMonitor:
    """Monitor cold start (model load + first inference)"""
    
    def __init__(self):
        self.load_start = None
        self.load_end = None
        self.first_inference_start = None
        self.first_inference_end = None
    
    def start_load(self):
        self.load_start = time.perf_counter()
    
    def end_load(self):
        self.load_end = time.perf_counter()
    
    def start_inference(self):
        self.first_inference_start = time.perf_counter()
    
    def end_inference(self):
        self.first_inference_end = time.perf_counter()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get cold start metrics"""
        if not all([self.load_start, self.load_end, self.first_inference_start, self.first_inference_end]):
            return {}
        
        load_time = (self.load_end - self.load_start) * 1000
        inference_time = (self.first_inference_end - self.first_inference_start) * 1000
        total_time = (self.first_inference_end - self.load_start) * 1000
        
        return {
            'model_load_time_ms': round(load_time, 2),
            'first_inference_time_ms': round(inference_time, 2),
            'total_cold_start_ms': round(total_time, 2)
        }
