#!/usr/bin/env python3
"""
Full Ollama Model Benchmark - Enhanced Version

Features:
- Dynamic model discovery from Ollama (no hardcoded models)
- CUDA/GPU monitoring integration
- Parallel execution using thread pools
- Human-readable text output files
- Resume capability for interrupted runs
- Per-model completion time tracking
- Extensive code documentation

Focus: Juniper/PyEZ/LangGraph/FastMCP use cases
"""

import requests
import time
import json
import os
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

# ============================================================================
# CUDA / GPU MONITORING
# ============================================================================

try:
    # pynvml is NVIDIA's Python bindings for NVML (NVIDIA Management Library)
    # It allows querying GPU status without needing PyTorch/TensorFlow
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetName
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: pynvml not installed. GPU monitoring disabled.")
    print("Install with: pip install nvidia-ml-py")


class GPUMonitor:
    """
    Monitors GPU utilization and memory during benchmark execution.
    Uses NVML (NVIDIA Management Library) for lightweight GPU querying.
    """
    
    def __init__(self):
        self.gpu_data = []
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
    def start(self):
        """Start GPU monitoring in a background thread."""
        if not CUDA_AVAILABLE:
            return
            
        try:
            nvmlInit()
            self.monitoring = True
            self.gpu_data = []
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
        except Exception as e:
            print(f"GPU monitoring init failed: {e}")
            
    def _monitor_loop(self):
        """Background thread that polls GPU stats every second."""
        while self.monitoring:
            try:
                stats = self._get_gpu_stats()
                with self._lock:
                    self.gpu_data.append(stats)
            except Exception:
                pass
            time.sleep(1)
            
    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Query current GPU statistics from all available GPUs."""
        stats = {"timestamp": datetime.now().isoformat(), "gpus": []}
        device_count = nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            name = nvmlDeviceGetName(handle)
            
            stats["gpus"].append({
                "id": i,
                "name": name.decode() if isinstance(name, bytes) else name,
                "memory_used_mb": mem_info.used // (1024 * 1024),
                "memory_total_mb": mem_info.total // (1024 * 1024),
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory
            })
        return stats
        
    def stop(self) -> List[Dict]:
        """Stop monitoring and return collected GPU data."""
        if not CUDA_AVAILABLE or not self.monitoring:
            return []
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
        try:
            nvmlShutdown()
        except:
            pass
            
        with self._lock:
            return self.gpu_data.copy()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ollama API endpoint - can be overridden via environment variable
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# State file for resume functionality
# Stores which models have been completed to enable resuming interrupted runs
STATE_FILE = os.path.expanduser("~/.benchmark_state.json")

# Default output directory for results
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/benchmark/results")

# ============================================================================
# BENCHMARK TASKS
# ============================================================================

# Each task defines a specific capability to test.
# Keywords are used for automated scoring (exact matches in response).
TASKS = [
    {
        "id": "speed_warmup",
        "name": "Speed: Warmup",
        "prompt": "Write a Python function that returns the sum of a list of integers.",
        "max_tokens": 256,
        "category": "speed",
        "score_keywords": ["def", "return", "sum"],
    },
    {
        "id": "speed_simple",
        "name": "Speed: Simple Code",
        "prompt": "Write a Python dataclass for a network interface with fields: name, ip, mac, status, speed_gbps. Include __post_init__ validation.",
        "max_tokens": 512,
        "category": "speed",
        "score_keywords": ["@dataclass", "def __post_init__", "name", "ip", "mac"],
    },
    {
        "id": "coding_pyez",
        "name": "Coding: PyEZ BGP",
        "prompt": """Write a Python async function using junos-eznc PyEZ that:
1. Connects to Juniper MX204 via NETCONF
2. Gets BGP neighbor summary via RPC
3. Returns List[TypedDict] with: neighbor_ip, peer_as, state, rx_prefixes
4. Handles ConnectError and RpcError separately with specific error messages
5. Uses async context manager, closes device in finally block
Return only working Python code with type hints.""",
        "max_tokens": 1024,
        "category": "coding",
        "score_keywords": [
            "async def", "Device(", "ConnectError", "RpcError",
            "TypedDict", "finally", "neighbor", "peer_as"
        ],
    },
    {
        "id": "coding_langgraph",
        "name": "Coding: LangGraph Node",
        "prompt": """Write a LangGraph StateGraph node for Juniper device onboarding:
- State TypedDict: device_ip, credentials, phase, interfaces, bgp_neighbors, errors
- Node validates: all interfaces UP, all BGP sessions Established
- Returns updated state with validation_passed: bool, error_details: list
- Uses PyEZ NETCONF, async def, proper typing
Return only working Python code.""",
        "max_tokens": 1024,
        "category": "coding",
        "score_keywords": [
            "TypedDict", "async def", "Device(", "StateGraph",
            "validation_passed", "Established", "interfaces", "errors"
        ],
    },
    {
        "id": "coding_fastmcp",
        "name": "Coding: FastMCP Tool",
        "prompt": """Write a FastMCP tool for Juniper route table lookup:
- Decorator: @mcp.tool()
- Params: device_ip: str, vrf: str = "default", prefix: str = None
- Uses PyEZ to fetch inet.0 or VRF route table via NETCONF
- Returns structured dict: {vrf, prefix_count, routes: [{prefix, protocol, next_hop, metric}]}
- Auth from env vars JUNOS_USER / JUNOS_PASS
- Proper MCP docstring for LLM consumption
Return only working Python code.""",
        "max_tokens": 1024,
        "category": "coding",
        "score_keywords": [
            "@mcp.tool", "device_ip", "vrf", "Device(",
            "environ", "next_hop", "protocol", "prefix_count"
        ],
    },
    {
        "id": "debug_pyez",
        "name": "Debug: Fix PyEZ Bugs",
        "prompt": """Find and fix ALL bugs in this PyEZ code. Comment each fix:
```python
async def get_mpls_lsp_stats(devices):
    results = []
    for ip in devices:
        dev = Device(ip, user=os.environ['USER'], password=os.environ['PASS'])
        dev.open()
        lsps = dev.rpc.get_mpls_lsp_information(detail=True)
        for lsp in lsps.findall('rsvp-session'):
            name = lsp.find('name').text
            bw = lsp.find('bandwidth').text
            results[ip] = {'lsp': name, 'bandwidth': bw}
        dev.close
    return results
```
Return fixed code only.""",
        "max_tokens": 1024,
        "category": "debug",
        "score_keywords": [
            "results = {}", "results[ip]", "dev.close()",
            "junos_user", "junos_pass", "append"
        ],
    },
    {
        "id": "reasoning_arch",
        "name": "Reasoning: Network Architecture",
        "prompt": """A Juniper MX204 running BGP shows intermittent route flaps every ~4 hours exactly.
BFD is enabled with 300ms intervals. MPLS LSPs are RSVP-TE with FRR.
What are the 3 most likely root causes, ranked by probability?
For each: explain the mechanism, how to confirm, and the fix.
Be specific to Junos CLI commands.""",
        "max_tokens": 1024,
        "category": "reasoning",
        "score_keywords": [
            "BFD", "RSVP", "timer", "show bgp", "show bfd",
            "hold-time", "graceful-restart", "rpd"
        ],
    },
]


# ============================================================================
# STATE MANAGEMENT (For Resume Functionality)
# ============================================================================

class BenchmarkState:
    """
    Manages persistent state to enable resuming interrupted benchmark runs.
    
    State is stored as JSON with:
    - completed_models: List of model names that finished
    - results: Partial results from completed models
    - timestamp: When the benchmark started
    - config: Benchmark configuration (tasks, runs, etc.)
    """
    
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = self._load()
        
    def _load(self) -> Dict:
        """Load state from disk if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "completed_models": [],
            "results": {},
            "timestamp": None,
            "config": {}
        }
        
    def save(self):
        """Persist current state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
            
    def is_completed(self, model: str) -> bool:
        """Check if a model has already been benchmarked."""
        return model in self.state.get("completed_models", [])
        
    def mark_completed(self, model: str, result: Dict):
        """Mark a model as completed and store its results."""
        if model not in self.state["completed_models"]:
            self.state["completed_models"].append(model)
        self.state["results"][model] = result
        self.save()
        
    def get_pending_models(self, all_models: List[str]) -> List[str]:
        """Filter to models that haven't been completed yet."""
        return [m for m in all_models if not self.is_completed(m)]
        
    def clear(self):
        """Clear all state - call after successful benchmark completion."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        self.state = {"completed_models": [], "results": {}, "timestamp": None, "config": {}}


# ============================================================================
# OLLAMA API INTERACTION
# ============================================================================

def get_available_models() -> List[Dict[str, Any]]:
    """
    Fetch all available models from Ollama API.
    
    Returns list of model dicts with:
    - name: Model identifier
    - params: Parameter size (e.g., "7B", "70B")
    - quant: Quantization level (e.g., "Q4_K_M")
    - size_gb: Model size in gigabytes
    - capabilities: List of capabilities (tools, vision, etc.)
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception as e:
        print(f"❌ Could not reach Ollama at {OLLAMA_URL}: {e}")
        return []
    
    # Enrich each model with detailed info via /api/show
    enriched_models = []
    for m in models:
        name = m["name"]
        size_gb = m.get("size", 0) / (1024**3)
        
        try:
            show_resp = requests.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": name},
                timeout=10
            ).json()
            
            details = show_resp.get("details", {})
            caps = show_resp.get("capabilities", [])
            
            enriched_models.append({
                "name": name,
                "params": details.get("parameter_size", "?"),
                "quant": details.get("quantization_level", "?"),
                "size_gb": round(size_gb, 1),
                "capabilities": caps,
                "has_tools": "tools" in caps,
                "family": details.get("family", "unknown"),
            })
        except Exception:
            # Fallback if /api/show fails
            enriched_models.append({
                "name": name,
                "params": "?",
                "quant": "?",
                "size_gb": round(size_gb, 1),
                "capabilities": [],
                "has_tools": None,
                "family": "unknown",
            })
            
    return enriched_models


def detect_tools(model_name: str) -> bool:
    """
    Check if a model supports tool calling via Ollama API.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model has 'tools' capability, False otherwise
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name},
            timeout=10
        )
        data = resp.json()
        return "tools" in data.get("capabilities", [])
    except Exception:
        return False


# ============================================================================
# TASK EXECUTION
# ============================================================================

def run_task(model: str, task: Dict, temperature: float = 0.1) -> Dict:
    """
    Execute a single benchmark task against Ollama API.
    
    Args:
        model: Model name to test
        task: Task dictionary with prompt, max_tokens, score_keywords
        temperature: Sampling temperature (lower = more deterministic)
        
    Returns:
        Dictionary with execution results:
        - status: "ok", "error", or "timeout"
        - tps: Tokens per second
        - score: Keyword match percentage
        - response: Model output text
        - tokens_in/out: Token counts
        - elapsed_sec: Total time
    """
    # Build the API payload for chat completion
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": task["max_tokens"],
            "num_ctx": 32768,  # Context window size
        }
    }
    
    start_time = time.time()
    
    try:
        # Send request to Ollama chat endpoint
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        data = resp.json()
        
        # Handle API errors
        if "error" in data:
            return {"status": "error", "error": data["error"], "tps": 0, "score": 0}
        
        # Calculate timing and token metrics
        elapsed = time.time() - start_time
        response_text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)  # Output tokens
        eval_duration = data.get("eval_duration", 1)  # Nanoseconds
        prompt_tokens = data.get("prompt_eval_count", 0)  # Input tokens
        
        # Tokens per second (eval_duration is in nanoseconds)
        tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0
        
        # Score based on keyword presence in response
        response_lower = response_text.lower()
        keywords_hit = [kw for kw in task["score_keywords"] if kw.lower() in response_lower]
        score = round(len(keywords_hit) / len(task["score_keywords"]) * 100, 1)
        
        return {
            "status": "ok",
            "elapsed_sec": round(elapsed, 1),
            "tps": round(tps, 1),
            "tokens_out": eval_count,
            "tokens_in": prompt_tokens,
            "score": score,
            "keywords_hit": keywords_hit,
            "keywords_missed": [kw for kw in task["score_keywords"] if kw.lower() not in response_lower],
            "response": response_text,
        }
        
    except requests.exceptions.Timeout:
        return {"status": "timeout", "tps": 0, "score": 0, "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e), "tps": 0, "score": 0}


# ============================================================================
# RESULT OUTPUT FORMATTING
# ============================================================================

def save_text_output(model_name: str, model_info: Dict, results: Dict, 
                     output_dir: str, completion_time: datetime):
    """
    Save model results to a human-readable text file.
    
    Creates a formatted text report with:
    - Model metadata
    - Completion timestamp
    - Per-task results with scores and responses
    - Summary statistics
    
    Args:
        model_name: Name of the model
        model_info: Model metadata dict
        results: Task results dict
        output_dir: Directory to save file
        completion_time: When the model finished benchmarking
    """
    # Sanitize model name for filename
    safe_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"{safe_name}_{completion_time.strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # Header section
        f.write("=" * 80 + "\n")
        f.write(f"BENCHMARK RESULTS FOR: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Model metadata
        f.write("MODEL METADATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Name:           {model_name}\n")
        f.write(f"Parameters:     {model_info.get('params', '?')}\n")
        f.write(f"Quantization:   {model_info.get('quant', '?')}\n")
        f.write(f"Size:           {model_info.get('size_gb', '?')} GB\n")
        f.write(f"Tool Support:   {'Yes' if model_info.get('has_tools') else 'No'}\n")
        f.write(f"Family:         {model_info.get('family', '?')}\n")
        f.write(f"Completed:      {completion_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        # Per-task results
        f.write("TASK RESULTS\n")
        f.write("-" * 40 + "\n\n")
        
        task_results = results.get("_tasks", {})
        for task_id, result in task_results.items():
            f.write(f"\n{'─' * 70}\n")
            f.write(f"Task: {task_id}\n")
            f.write(f"Status: {result.get('status', 'unknown')}\n")
            
            if result.get('status') == 'ok':
                f.write(f"Score: {result.get('score', 0)}%\n")
                f.write(f"TPS: {result.get('tps', 0)} tokens/sec\n")
                f.write(f"Time: {result.get('elapsed_sec', 0)}s\n")
                f.write(f"Tokens: {result.get('tokens_out', 0)} out, {result.get('tokens_in', 0)} in\n")
                
                if result.get('keywords_hit'):
                    f.write(f"Keywords Hit: {', '.join(result['keywords_hit'])}\n")
                if result.get('keywords_missed'):
                    f.write(f"Keywords Missed: {', '.join(result['keywords_missed'])}\n")
                    
                f.write("\n--- RESPONSE ---\n")
                f.write(result.get('response', '[No response]'))
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
            f.write("\n")
        
        # Summary section
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        summary = results.get("_summary", {})
        if summary:
            f.write(f"Average Score:   {summary.get('avg_score', 0)}%\n")
            f.write(f"Average TPS:     {summary.get('avg_tps', 0)} tok/s\n")
            f.write(f"Tasks Completed: {summary.get('tasks_ok', 0)}/{summary.get('tasks_total', 0)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
    return filepath


# ============================================================================
# BENCHMARK ORCHESTRATION
# ============================================================================

def benchmark_single_model(model: Dict, tasks: List[Dict], runs: int, 
                           output_dir: str, state: BenchmarkState) -> Dict:
    """
    Run complete benchmark for a single model.
    
    This function is designed to be run in a thread pool for parallel execution.
    It handles all tasks for one model and saves progress incrementally.
    
    Args:
        model: Model metadata dict
        tasks: List of task dicts to run
        runs: Number of runs per task (for averaging)
        output_dir: Where to save output files
        state: BenchmarkState for resume tracking
        
    Returns:
        Complete results dict for the model
    """
    model_name = model["name"]
    start_time = time.time()
    
    # Print model header (thread-safe with lock)
    print(f"\n🤖 {model_name}  [{model.get('params','?')} {model.get('quant','?')}]")
    print(f"   Tools: {'✅' if model.get('has_tools') else '❌'}  |  Family: {model.get('family','?')}")
    print("-" * 65)
    
    # Initialize results structure
    model_results = {
        "_info": model,
        "_tasks": {},
        "_start_time": datetime.now().isoformat()
    }
    
    scores = []
    tps_values = []
    
    # Execute each task
    for task in tasks:
        print(f"  ⏳ {task['name']:<35}", end="", flush=True)
        
        # Run task multiple times if requested
        run_results = []
        for _ in range(runs):
            result = run_task(model_name, task)
            run_results.append(result)
            
        # Average results across multiple runs
        ok_runs = [r for r in run_results if r["status"] == "ok"]
        if ok_runs:
            avg_tps = sum(r["tps"] for r in ok_runs) / len(ok_runs)
            avg_score = sum(r["score"] for r in ok_runs) / len(ok_runs)
            best = max(ok_runs, key=lambda r: r["score"])
            result = {**best, "tps": avg_tps, "score": avg_score}
            
            scores.append(avg_score)
            tps_values.append(avg_tps)
            
            print(f" ✅ {avg_tps:>6.1f} tok/s | score={avg_score:>5.1f}% | {best['elapsed_sec']}s")
            if best.get("keywords_missed"):
                print(f"     missed: {best['keywords_missed']}")
        else:
            # All runs failed
            result = run_results[-1]
            print(f" ❌ {result.get('status')} {result.get('error','')[:50]}")
            
        model_results["_tasks"][task["id"]] = result
        
    # Calculate summary statistics
    if scores:
        avg_score = sum(scores) / len(scores)
        avg_tps = sum(tps_values) / len(tps_values)
        model_results["_summary"] = {
            "avg_score": round(avg_score, 1),
            "avg_tps": round(avg_tps, 1),
            "tasks_ok": len(scores),
            "tasks_total": len(tasks),
        }
        print(f"  {'─'*60}")
        print(f"  Summary: avg_score={avg_score:.1f}%  avg_tps={avg_tps:.1f}  tasks={len(scores)}/{len(tasks)}")
        
    # Record completion
    completion_time = datetime.now()
    elapsed = time.time() - start_time
    model_results["_completion_time"] = completion_time.isoformat()
    model_results["_total_duration_sec"] = round(elapsed, 1)
    
    print(f"  Completed at: {completion_time.strftime('%H:%M:%S')} (took {elapsed:.1f}s)")
    
    # Save human-readable text output
    text_file = save_text_output(model_name, model, model_results, output_dir, completion_time)
    print(f"  📝 Text output: {text_file}")
    
    # Update state for resume capability
    state.mark_completed(model_name, model_results)
    
    return model_results


def run_benchmark(
    selected_models: List[str] = None,
    selected_tasks: List[Dict] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    runs: int = 1,
    max_workers: int = 2,
    resume: bool = False,
):
    """
    Main benchmark orchestration function.
    
    Manages the complete benchmark workflow:
    1. Discovers available models from Ollama
    2. Filters based on user selection
    3. Runs benchmarks in parallel thread pool
    4. Generates leaderboard and reports
    5. Handles resume functionality
    
    Args:
        selected_models: Specific model names to test (None = all)
        selected_tasks: Specific tasks to run (None = all)
        output_dir: Directory for output files
        runs: Number of runs per task
        max_workers: Max parallel threads for concurrent model testing
        resume: If True, skip already-completed models from previous run
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize state manager for resume functionality
    state = BenchmarkState()
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fetch available models from Ollama (dynamic discovery, no hardcoding)
    print("\n" + "=" * 80)
    print("  FETCHING MODELS FROM OLLAMA")
    print("=" * 80)
    
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available. Exiting.")
        return None, None
        
    # Display discovered models
    print(f"\n  Found {len(available_models)} models:\n")
    print(f"  {'Name':<35} {'Params':>10} {'Quant':>10} {'Size':>8} {'Tools'}")
    print("  " + "─" * 75)
    for m in sorted(available_models, key=lambda x: x["name"]):
        tools = "✅" if m.get("has_tools") else "❌"
        print(f"  {m['name']:<35} {m['params']:>10} {m['quant']:>10} {m['size_gb']:>7.1f}G {tools}")
    
    # Filter models based on selection
    if selected_models:
        models_to_test = [m for m in available_models if m["name"] in selected_models]
    else:
        models_to_test = available_models
        
    # Apply resume filter if requested
    if resume:
        pending = state.get_pending_models([m["name"] for m in models_to_test])
        skipped = [m["name"] for m in models_to_test if m["name"] not in pending]
        if skipped:
            print(f"\n  ⏭️  Resuming: skipping {len(skipped)} already-completed models: {skipped}")
        models_to_test = [m for m in models_to_test if m["name"] in pending]
        
        # Load previous results if resuming
        all_results = state.state.get("results", {})
    else:
        # Fresh run - clear previous state
        state.clear()
        state.state["timestamp"] = timestamp
        all_results = {}
        
    if not models_to_test:
        print("\n  ✅ All models already completed!")
        gpu_monitor.stop()
        return all_results, []
        
    # Task selection
    tasks_to_test = selected_tasks or TASKS
    
    # Print benchmark configuration
    print(f"\n{'='*65}")
    print(f"  Benchmark Configuration")
    print(f"{'='*65}")
    print(f"  Models:    {len(models_to_test)} (parallel workers: {max_workers})")
    print(f"  Tasks:     {len(tasks_to_test)}")
    print(f"  Runs:      {runs}")
    print(f"  Output:    {output_dir}")
    print(f"  Resume:    {'Yes' if resume else 'No (use --resume to continue interrupted runs)'}")
    print(f"{'='*65}\n")
    
    # Execute benchmarks using thread pool for parallel execution
    # Each model runs in its own thread, allowing concurrent testing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all model benchmarks to the thread pool
        future_to_model = {
            executor.submit(
                benchmark_single_model, 
                model, 
                tasks_to_test, 
                runs, 
                output_dir,
                state
            ): model 
            for model in models_to_test
        }
        
        # Collect results as threads complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                all_results[model["name"]] = result
            except Exception as e:
                print(f"\n  ❌ Error benchmarking {model['name']}: {e}")
                
    # Stop GPU monitoring and get stats
    gpu_data = gpu_monitor.stop()
    if gpu_data:
        print(f"\n  📊 GPU monitoring: collected {len(gpu_data)} samples")
        
    # Generate leaderboard from all results (including resumed ones)
    print(f"\n{'='*65}")
    print(f"  LEADERBOARD")
    print(f"{'='*65}")
    print(f"{'Model':<32} {'Params':>8} {'Quant':>8} {'Score':>7} {'tok/s':>7} {'Tools'}")
    print(f"{'─'*65}")
    
    # Sort models by average score
    ranked = sorted(
        [(m, r) for m, r in all_results.items() if "_summary" in r],
        key=lambda x: x[1]["_summary"]["avg_score"],
        reverse=True
    )
    
    medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ ", "6️⃣ ", "7️⃣ ", "8️⃣ ", "9️⃣ ", "🔟"]
    for i, (model_name, result) in enumerate(ranked):
        summary = result["_summary"]
        info = result["_info"]
        medal = medals[i] if i < len(medals) else f"{i+1}."
        tools = "✅" if info.get("has_tools") else "❌"
        name = model_name.split(":")[0][:31]
        print(f"{medal} {name:<30} {info.get('params','?'):>8} {info.get('quant','?'):>8} "
              f"{summary['avg_score']:>6.1f}% {summary['avg_tps']:>6.1f} {tools}")
    
    # Per-category breakdown
    print(f"\n{'='*65}")
    print(f"  SCORES BY CATEGORY")
    print(f"{'='*65}")
    categories = list(dict.fromkeys(t["category"] for t in tasks_to_test))
    header = f"{'Model':<28}" + "".join(f"{c:>12}" for c in categories)
    print(header)
    print("─" * 65)
    
    for model_name, result in ranked:
        row = f"{model_name.split(':')[0][:27]:<28}"
        for cat in categories:
            cat_tasks = [t["id"] for t in tasks_to_test if t["category"] == cat]
            cat_scores = [
                result["_tasks"][tid]["score"]
                for tid in cat_tasks
                if tid in result["_tasks"] and result["_tasks"][tid].get("status") == "ok"
            ]
            avg = sum(cat_scores) / len(cat_scores) if cat_scores else 0
            row += f"{avg:>11.1f}%"
        print(row)
    
    # Save structured JSON results
    slim_results = {
        m: {
            "_info": r["_info"],
            "_summary": r.get("_summary", {}),
            "_completion_time": r.get("_completion_time"),
            "_total_duration_sec": r.get("_total_duration_sec"),
            "_tasks": {
                tid: {k: v for k, v in t.items() if k != "response"}
                for tid, t in r["_tasks"].items()
            }
        }
        for m, r in all_results.items()
    }
    
    json_file = f"{output_dir}/benchmark_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(slim_results, f, indent=2)
    
    # Save full responses separately
    full_file = f"{output_dir}/responses_{timestamp}.json"
    with open(full_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save GPU stats if available
    if gpu_data:
        gpu_file = f"{output_dir}/gpu_stats_{timestamp}.json"
        with open(gpu_file, "w") as f:
            json.dump(gpu_data, f, indent=2)
        print(f"\n  📊 GPU stats:   {gpu_file}")
    
    print(f"\n  💾 JSON results: {json_file}")
    print(f"  📝 Full responses: {full_file}")
    print(f"\n  📁 Text outputs in: {output_dir}")
    
    # Clear state on successful completion
    if not resume or len(state.get_pending_models([m["name"] for m in available_models])) == 0:
        state.clear()
        print("\n  ✅ Benchmark complete!")
    else:
        print(f"\n  ⚠️  Benchmark incomplete. Run with --resume to continue.")
    
    return all_results, ranked


def review_responses(results_file: str, model_filter: str = None, task_filter: str = None):
    """
    Interactive TUI for reviewing model responses.
    
    Allows browsing through saved responses with keyboard navigation:
    - Enter: next response
    - q: quit
    - s: skip to next model
    
    Args:
        results_file: Path to responses JSON file
        model_filter: Optional substring to filter models
        task_filter: Optional task ID to filter tasks
    """
    with open(results_file) as f:
        data = json.load(f)
    
    for model, result in data.items():
        if model_filter and model_filter not in model:
            continue
        for task_id, task_result in result.get("_tasks", {}).items():
            if task_filter and task_filter != task_id:
                continue
            if "response" not in task_result:
                continue
                
            print(f"\n{'='*65}")
            print(f"MODEL : {model}")
            print(f"TASK  : {task_id}")
            print(f"SCORE : {task_result.get('score')}%  |  {task_result.get('tps')} tok/s")
            print(f"{'='*65}")
            print(task_result["response"])
            
            cmd = input("\n[Enter=next | q=quit | s=skip model] → ").strip()
            if cmd == "q":
                return
            if cmd == "s":
                break


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ollama Model Benchmark - Enhanced with CUDA, Parallel Execution, and Resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on all available models
  python benchmark-individual-llm.py
  
  # Test specific models with 3 runs per task
  python benchmark-individual-llm.py --models qwen2.5-coder:32b gemma4:31b --runs 3
  
  # Resume interrupted benchmark
  python benchmark-individual-llm.py --resume
  
  # Quick speed test only, 4 parallel workers
  python benchmark-individual-llm.py --quick --workers 4
  
  # Show available models and exit
  python benchmark-individual-llm.py --inventory
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific model names to test (default: all available)"
    )
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        help="Specific task IDs to run (default: all)"
    )
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run speed tasks only"
    )
    parser.add_argument(
        "--coding", 
        action="store_true", 
        help="Run coding and debug tasks only"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=1, 
        help="Number of runs per task (for averaging results)"
    )
    parser.add_argument(
        "--review", 
        type=str, 
        help="Review responses from a JSON results file"
    )
    parser.add_argument(
        "--inventory", 
        action="store_true", 
        help="Show Ollama model inventory and exit"
    )
    parser.add_argument(
        "--output", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=2,
        help="Max parallel workers for concurrent model testing (default: 2)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume interrupted benchmark from where it left off"
    )
    
    args = parser.parse_args()
    
    # Handle inventory display
    if args.inventory:
        models = get_available_models()
        if models:
            print(f"\n{'='*80}")
            print(f"  AVAILABLE OLLAMA MODELS")
            print(f"{'='*80}")
            print(f"  {'Name':<35} {'Params':>10} {'Quant':>10} {'Size':>9} {'Capabilities'}")
            print(f"  {'─'*75}")
            for m in sorted(models, key=lambda x: x["name"]):
                caps = ", ".join(m.get("capabilities", ["completion"]))
                print(f"  {m['name']:<35} {m['params']:>10} {m['quant']:>10} "
                      f"{m['size_gb']:>8.1f}G {caps}")
        else:
            print("❌ No models found or Ollama not reachable.")
        sys.exit(0)
    
    # Handle response review mode
    if args.review:
        review_responses(args.review)
        sys.exit(0)
    
    # Select tasks based on arguments
    selected_tasks = TASKS
    if args.quick:
        selected_tasks = [t for t in TASKS if t["category"] == "speed"]
    elif args.coding:
        selected_tasks = [t for t in TASKS if t["category"] in ("coding", "debug")]
    elif args.tasks:
        selected_tasks = [t for t in TASKS if t["id"] in args.tasks]
    
    # Run the benchmark
    run_benchmark(
        selected_models=args.models,
        selected_tasks=selected_tasks,
        output_dir=args.output,
        runs=args.runs,
        max_workers=args.workers,
        resume=args.resume,
    )
