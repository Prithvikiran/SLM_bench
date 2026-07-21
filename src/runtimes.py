"""
Runtime Implementations
Abstracts four inference runtimes: llama.cpp, Ollama, PyTorch, ONNX Runtime
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import subprocess
import requests
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

class InferenceRuntime(ABC):
    """Base class for inference runtimes"""
    
    def __init__(self, model_name: str, quantization: str = "fp16"):
        self.model_name = model_name
        self.quantization = quantization
        self.loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load model into memory"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 64, 
                temperature: float = 0.1, top_p: float = 0.9) -> Tuple[str, int]:
        """
        Generate text
        
        Returns:
            (generated_text, num_tokens)
        """
        pass
    
    @abstractmethod
    def unload_model(self):
        """Unload model from memory"""
        pass


class LlamaCppRuntime(InferenceRuntime):
    """llama.cpp inference via Python bindings"""
    
    def load_model(self) -> bool:
        """Load GGUF model via llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            model_path = self._get_gguf_path()
            if not os.path.exists(model_path):
                logger.error(f"GGUF model not found: {model_path}")
                return False
            
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Use GPU
                n_ctx=2048,
                verbose=False
            )
            self.loaded = True
            logger.info(f"Loaded {self.model_name} via llama.cpp")
            return True
        
        except ImportError:
            logger.error("llama-cpp-python not installed: pip install llama-cpp-python")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 64,
                temperature: float = 0.1, top_p: float = 0.9) -> Tuple[str, int]:
        """Generate using llama.cpp"""
        if not self.loaded:
            return "", 0
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False
        )
        
        text = response['choices'][0]['text']
        tokens = response['usage']['completion_tokens']
        
        return text, tokens
    
    def unload_model(self):
        """Cleanup"""
        if hasattr(self, 'llm'):
            del self.llm
        self.loaded = False
    
    def _get_gguf_path(self) -> str:
        """Construct path to GGUF model file"""
        quant_map = {
            'q4': 'Q4_K_M',
            'q5': 'Q5_K_M',
            'fp16': 'fp16'
        }
        quant = quant_map.get(self.quantization, 'Q4_K_M')
        return f"./models/{self.model_name}-{quant}.gguf"


class OllamaRuntime(InferenceRuntime):
    """Ollama inference via HTTP API"""
    
    # (connect timeout sec, read timeout sec) — avoids spurious TimeoutError on slow pulls/generation
    _TIMEOUT_SHORT = (30, 120)
    _TIMEOUT_LONG = (30, 900)
    
    @classmethod
    def _base_url(cls) -> str:
        raw = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
        if not raw.lower().startswith(("http://", "https://")):
            raw = f"http://{raw}"
        return raw
    
    def _tags_list(self) -> list:
        r = requests.get(f"{self._base_url()}/api/tags", timeout=self._TIMEOUT_SHORT)
        r.raise_for_status()
        return r.json().get("models") or []
    
    def _model_is_local(self) -> bool:
        """True if this model name is already in `ollama list`."""
        try:
            want = self.model_name.strip()
            for m in self._tags_list():
                name = (m.get("name") or "").strip()
                if name == want or name.startswith(want + ":"):
                    return True
        except requests.RequestException:
            return False
        return False
    
    def load_model(self) -> bool:
        """Ensure model is available (skip pull if already local)."""
        try:
            response = requests.get(
                f"{self._base_url()}/api/tags",
                timeout=self._TIMEOUT_SHORT,
            )
            if response.status_code != 200:
                logger.error(
                    f"Ollama not reachable at {self._base_url()} (HTTP {response.status_code})"
                )
                return False
        except requests.RequestException as e:
            logger.error(
                f"Ollama not reachable at {self._base_url()}: {e}. "
                "Start Ollama and try again (or set OLLAMA_HOST)."
            )
            return False
        
        try:
            if self._model_is_local():
                logger.info(f"Ollama model already present: {self.model_name}")
                self.loaded = True
                return True
            
            logger.info(f"Pulling {self.model_name} via Ollama (first time may take a while)...")
            pull_response = requests.post(
                f"{self._base_url()}/api/pull",
                json={"name": self.model_name, "stream": False},
                timeout=self._TIMEOUT_LONG,
            )
            if pull_response.status_code != 200:
                logger.error(
                    f"Ollama pull failed ({pull_response.status_code}): {pull_response.text[:500]}"
                )
                logger.error(
                    "If you see 'file does not exist', the tag is wrong. "
                    "Check https://ollama.com/library and update ollama_model in config.json."
                )
                return False
            
            self.loaded = True
            logger.info(f"Loaded {self.model_name} via Ollama")
            return True
        
        except Exception as e:
            logger.error(f"Ollama load failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 64,
                temperature: float = 0.1, top_p: float = 0.9) -> Tuple[str, int]:
        """Generate using Ollama API"""
        if not self.loaded:
            return "", 0
        
        try:
            response = requests.post(
                f"{self._base_url()}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                },
                timeout=self._TIMEOUT_LONG,
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['response'], len(data['response'].split())
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return "", 0
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "", 0
    
    def unload_model(self):
        """Clear local loaded flag only.

        Ollama keeps weights in memory; calling /api/delete would *remove* model
        files from disk. We do not delete the user's models between trials.
        """
        self.loaded = False


class PyTorchRuntime(InferenceRuntime):
    """PyTorch + HuggingFace Transformers inference"""
    
    def load_model(self) -> bool:
        """Load model via HuggingFace Transformers"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load quantization if needed
            if self.quantization == "int8":
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quant_config
                )
            elif self.quantization == "q4":
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quant_config
                )
            else:  # fp16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.loaded = True
            logger.info(f"Loaded {self.model_name} via PyTorch")
            return True
        
        except Exception as e:
            logger.error(f"PyTorch load failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 64,
                temperature: float = 0.1, top_p: float = 0.9) -> Tuple[str, int]:
        """Generate using PyTorch"""
        if not self.loaded:
            return "", 0
        
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part
            text = text[len(prompt):]
            
            tokens = len(self.tokenizer.encode(text))
            return text, tokens
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "", 0
    
    def unload_model(self):
        """Cleanup"""
        if hasattr(self, 'model'):
            del self.model
            import torch
            torch.cuda.empty_cache()
        self.loaded = False


class ONNXRuntime(InferenceRuntime):
    """ONNX Runtime inference (optimized for Windows)"""
    
    def load_model(self) -> bool:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            onnx_path = self._get_onnx_path()
            if not os.path.exists(onnx_path):
                logger.error(
                    f"ONNX model file missing for quantization '{self.quantization}': {onnx_path}"
                )
                logger.error(
                    "Export or place the ONNX file at this path, or remove this "
                    "quantization from config.json for onnx_runtime."
                )
                return False
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use CPU or GPU provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            self.loaded = True
            logger.info(f"Loaded {self.model_name} via ONNX Runtime")
            return True
        
        except Exception as e:
            logger.error(f"ONNX load failed: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 64,
                temperature: float = 0.1, top_p: float = 0.9) -> Tuple[str, int]:
        """Generate using ONNX Runtime"""
        if not self.loaded:
            return "", 0
        
        # Note: ONNX Runtime doesn't have built-in generation
        # This would require implementing autoregressive generation
        # For now, this is a placeholder
        logger.warning("Full ONNX generation requires additional implementation")
        return "", 0
    
    def unload_model(self):
        """Cleanup"""
        if hasattr(self, 'session'):
            del self.session
        self.loaded = False
    
    def _get_onnx_path(self) -> str:
        """Get path to ONNX model"""
        return f"./models/{self.model_name}-{self.quantization}.onnx"


class RuntimeFactory:
    """Factory to create appropriate runtime"""
    
    RUNTIMES = {
        'llama_cpp': LlamaCppRuntime,
        'ollama': OllamaRuntime,
        'pytorch': PyTorchRuntime,
        'onnx_runtime': ONNXRuntime
    }
    
    @classmethod
    def create(cls, runtime_name: str, model_name: str, 
              quantization: str = "fp16") -> Optional[InferenceRuntime]:
        """Create a runtime instance"""
        if runtime_name not in cls.RUNTIMES:
            logger.error(f"Unknown runtime: {runtime_name}")
            return None
        
        RuntimeClass = cls.RUNTIMES[runtime_name]
        return RuntimeClass(model_name, quantization)
