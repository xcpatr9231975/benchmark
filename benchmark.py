#!/usr/bin/env python3
"""
Full Ollama Model Benchmark
- Tools-capable models: tested via Claude Code style agentic prompts
- No-tools models (qwen35-claude46-q8): tested via direct Ollama API
- Focuses on Juniper/PyEZ/LangGraph/FastMCP use cases
Save as: ~/benchmark/run_benchmark.py
"""
import requests, time, json, os, sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

OLLAMA_URL = "http://localhost:11434"

# ── Model registry ──────────────────────────────────────────
MODELS = {
    "qwen35-claude46-q8:latest": {
        "has_tools": False,
        "params": "26.9B", "quant": "Q8_0", "arch": "qwen35"
    },
    "qwen3-coder-next:latest": {
        "has_tools": True,
        "params": "79.7B", "quant": "Q4_K_M", "arch": "qwen3next"
    },
    "gpt-oss:120b": {
        "has_tools": None,  # unknown, will auto-detect
        "params": "116.8B", "quant": "MXFP4", "arch": "gptoss"
    },
    "qwen2.5-coder:32b": {
        "has_tools": None,
        "params": "32.8B", "quant": "Q4_K_M", "arch": "qwen2"
    },
    "glm-4.7-flash:latest": {
        "has_tools": None,
        "params": "29.9B", "quant": "Q4_K_M", "arch": "glm4moelite"
    },
    "gemma4:31b": {
        "has_tools": None,
        "params": "31B", "quant": "Q4_K_M", "arch": "gemma4"
    },
    "deepseek-coder-v2:16b": {
        "has_tools": None,
        "params": "15.7B", "quant": "Q4_0", "arch": "deepseek2"
    },
    "nemotron:latest": {
        "has_tools": None,
        "params": "70.6B", "quant": "Q4_K_M", "arch": "llama"
    },
}

# ── Benchmark tasks ──────────────────────────────────────────
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


def show_ollama_models():
    """Display capabilities and parameters for all models available in Ollama"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = resp.json().get("models", [])
    except Exception as e:
        print(f"  ❌ Could not reach Ollama: {e}")
        return

    if not models:
        print("  No models found in Ollama.")
        return

    print(f"\n{'='*90}")
    print(f"  OLLAMA MODEL INVENTORY")
    print(f"{'='*90}")
    print(f"  {'Model':<35} {'Params':>10} {'Quant':>10} {'Size':>9} {'Capabilities'}")
    print(f"  {'─'*85}")

    for m in sorted(models, key=lambda x: x["name"]):
        name = m["name"]
        size_gb = m.get("size", 0) / (1024**3)
        try:
            show = requests.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": name},
                timeout=10
            ).json()
            details = show.get("details", {})
            params = details.get("parameter_size", "?")
            quant = details.get("quantization_level", "?")
            caps = show.get("capabilities", [])
            caps_str = ", ".join(caps) if caps else "completion"
        except:
            params, quant, caps_str = "?", "?", "?"

        print(f"  {name:<35} {params:>10} {quant:>10} {size_gb:>8.1f}G {caps_str}")

    print(f"  {'─'*85}")
    print(f"  Total: {len(models)} models\n")


def detect_tools(model_name: str) -> bool:
    """Auto-detect if model supports tools via Ollama API"""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": model_name},
            timeout=10
        )
        data = resp.json()
        caps = data.get("capabilities", [])
        return "tools" in caps
    except:
        return False


def run_task(model: str, task: dict, temperature: float = 0.1) -> dict:
    """Run a single benchmark task against Ollama API"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": task["max_tokens"],
            "num_ctx": 32768,
        }
    }

    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=300
        )
        data = resp.json()

        if "error" in data:
            return {"status": "error", "error": data["error"], "tps": 0}

        elapsed = time.time() - start
        response = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1)
        prompt_tokens = data.get("prompt_eval_count", 0)
        tps = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

        # Score based on keyword presence
        r = response.lower()
        keywords_hit = [kw for kw in task["score_keywords"] if kw.lower() in r]
        score = round(len(keywords_hit) / len(task["score_keywords"]) * 100, 1)

        return {
            "status": "ok",
            "elapsed_sec": round(elapsed, 1),
            "tps": round(tps, 1),
            "tokens_out": eval_count,
            "tokens_in": prompt_tokens,
            "score": score,
            "keywords_hit": keywords_hit,
            "keywords_missed": [kw for kw in task["score_keywords"] if kw.lower() not in r],
            "response": response,
        }

    except requests.exceptions.Timeout:
        return {"status": "timeout", "tps": 0, "score": 0}
    except Exception as e:
        return {"status": "error", "error": str(e), "tps": 0, "score": 0}


def run_benchmark(
    selected_models: list = None,
    selected_tasks: list = None,
    output_dir: str = os.path.expanduser("~/benchmark/results"),
    runs: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Show inventory of all available Ollama models
    show_ollama_models()

    models_to_test = selected_models or list(MODELS.keys())
    tasks_to_test  = selected_tasks  or TASKS

    # Auto-detect tool support
    print("\n🔍 Detecting tool support...")
    for model in models_to_test:
        if model in MODELS and MODELS[model]["has_tools"] is None:
            MODELS[model]["has_tools"] = detect_tools(model)
        status = "✅ tools" if MODELS.get(model, {}).get("has_tools") else "❌ no tools"
        print(f"   {model:<40} {status}")

    all_results = {}
    print(f"\n{'='*65}")
    print(f"  Benchmark Start: {timestamp}")
    print(f"  Models: {len(models_to_test)} | Tasks: {len(tasks_to_test)} | Runs: {runs}")
    print(f"{'='*65}")

    for model in models_to_test:
        info = MODELS.get(model, {})
        print(f"\n🤖 {model}  [{info.get('params','?')} {info.get('quant','?')}]")
        print(f"   Tools: {'✅' if info.get('has_tools') else '❌'}  |  Arch: {info.get('arch','?')}")
        print("-" * 65)

        all_results[model] = {"_info": info, "_tasks": {}}
        model_scores = []
        model_tps = []

        for task in tasks_to_test:
            print(f"  ⏳ {task['name']:<35}", end="", flush=True)

            # Run multiple times if requested
            run_results = []
            for r in range(runs):
                result = run_task(model, task)
                run_results.append(result)

            # Average runs
            ok_runs = [r for r in run_results if r["status"] == "ok"]
            if ok_runs:
                avg_tps   = sum(r["tps"] for r in ok_runs) / len(ok_runs)
                avg_score = sum(r["score"] for r in ok_runs) / len(ok_runs)
                best = max(ok_runs, key=lambda r: r["score"])
                result = {**best, "tps": avg_tps, "score": avg_score}
                model_scores.append(avg_score)
                model_tps.append(avg_tps)
                print(f" ✅ {avg_tps:>6.1f} tok/s | score={avg_score:>5.1f}% | {best['elapsed_sec']}s")
                if best.get("keywords_missed"):
                    print(f"     missed: {best['keywords_missed']}")
            else:
                result = run_results[-1]
                print(f" ❌ {result.get('status')} {result.get('error','')[:50]}")

            all_results[model]["_tasks"][task["id"]] = result

        # Model summary
        if model_scores:
            avg_s = sum(model_scores) / len(model_scores)
            avg_t = sum(model_tps) / len(model_tps)
            all_results[model]["_summary"] = {
                "avg_score": round(avg_s, 1),
                "avg_tps": round(avg_t, 1),
                "tasks_ok": len(model_scores),
                "tasks_total": len(tasks_to_test),
            }
            print(f"  {'─'*60}")
            print(f"  Summary: avg_score={avg_s:.1f}%  avg_tps={avg_t:.1f}  tasks={len(model_scores)}/{len(tasks_to_test)}")

    # ── Leaderboard ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  LEADERBOARD")
    print(f"{'='*65}")
    print(f"{'Model':<32} {'Params':>8} {'Quant':>8} {'Score':>7} {'tok/s':>7} {'Tools'}")
    print(f"{'─'*65}")

    ranked = sorted(
        [(m, r) for m, r in all_results.items() if "_summary" in r],
        key=lambda x: x[1]["_summary"]["avg_score"],
        reverse=True
    )

    medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ ", "6️⃣ ", "7️⃣ "]
    for i, (model, result) in enumerate(ranked):
        s    = result["_summary"]
        info = result["_info"]
        medal = medals[i] if i < len(medals) else f"{i+1}."
        tools = "✅" if info.get("has_tools") else "❌"
        name  = model.split(":")[0][-31:]
        print(f"{medal} {name:<30} {info.get('params','?'):>8} {info.get('quant','?'):>8} "
              f"{s['avg_score']:>6.1f}% {s['avg_tps']:>6.1f} {tools}")

    # ── Per-category breakdown ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SCORES BY CATEGORY")
    print(f"{'='*65}")
    categories = list(dict.fromkeys(t["category"] for t in tasks_to_test))
    header = f"{'Model':<28}" + "".join(f"{c:>12}" for c in categories)
    print(header)
    print("─" * 65)

    for model, result in ranked:
        row = f"{model.split(':')[0][-27:]:<28}"
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

    # ── Save results ──────────────────────────────────────────
    out_file = f"{output_dir}/benchmark_{timestamp}.json"
    slim = {
        m: {
            "_info": r["_info"],
            "_summary": r.get("_summary", {}),
            "_tasks": {
                tid: {k: v for k, v in t.items() if k != "response"}
                for tid, t in r["_tasks"].items()
            }
        }
        for m, r in all_results.items()
    }
    with open(out_file, "w") as f:
        json.dump(slim, f, indent=2)

    # Save full responses separately
    full_file = f"{output_dir}/responses_{timestamp}.json"
    with open(full_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Results:   {out_file}")
    print(f"📝 Responses: {full_file}")

    return all_results, ranked


def review_responses(results_file: str, model_filter: str = None, task_filter: str = None):
    """Interactive response reviewer"""
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ollama Model Benchmark")
    parser.add_argument("--models",   nargs="+", help="Models to test")
    parser.add_argument("--tasks",    nargs="+", help="Task IDs to run")
    parser.add_argument("--quick",    action="store_true", help="Speed tasks only")
    parser.add_argument("--coding",   action="store_true", help="Coding tasks only")
    parser.add_argument("--runs",     type=int, default=1, help="Runs per task")
    parser.add_argument("--review",   type=str, help="Review responses from file")
    parser.add_argument("--inventory", action="store_true", help="Show Ollama model inventory and exit")
    parser.add_argument("--output",   default=os.path.expanduser("~/benchmark/results"))
    args = parser.parse_args()

    if args.inventory:
        show_ollama_models()
        sys.exit(0)

    if args.review:
        review_responses(args.review)
        sys.exit(0)

    selected_tasks = TASKS
    if args.quick:
        selected_tasks = [t for t in TASKS if t["category"] == "speed"]
    elif args.coding:
        selected_tasks = [t for t in TASKS if t["category"] in ("coding", "debug")]
    elif args.tasks:
        selected_tasks = [t for t in TASKS if t["id"] in args.tasks]

    run_benchmark(
        selected_models=args.models,
        selected_tasks=selected_tasks,
        output_dir=args.output,
        runs=args.runs,
    )
