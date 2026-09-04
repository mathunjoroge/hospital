#!/usr/bin/env python3
"""
Terminal Agent – NVIDIA Free Models + optional gstack
Usage:
  python agent.py
  python agent.py --task "create a flask API"
  python agent.py --review myfile.py
  python agent.py --benchmark              # test all models for efficiency
  python agent.py --benchmark --task "write a bubble sort function"
"""

import os, sys, json, re, argparse, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ----------------------------------------------------------------------
# Config
WORKSPACE = Path.home() / "nvidia_agent_workspace"
WORKSPACE.mkdir(exist_ok=True)

# Current working models on integrate.api.nvidia.com (as of July 2026)
# Dead/removed: phi-3-mini-4k, gemma-2b-it, mixtral-8x7b, mistral-7b-v0.3, nemotron-70b
ALL_MODELS = [
    # Fast small models
    "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct",
    "nvidia/nemotron-mini-4b-instruct",
    # Mid-tier
    "meta/llama-3.1-8b-instruct",
    # NVIDIA Nemotron Nano family (MoE – 30B params, ~3B active = efficient)
    "nvidia/nemotron-3-nano-30b-a3b",                  # text, coding, reasoning
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",   # multimodal + reasoning
    # Large
    "meta/llama-3.1-70b-instruct",
]

# Default role assignments
PLANNER = "meta/llama-3.2-1b-instruct"
CODER   = "nvidia/nemotron-3-nano-30b-a3b"   # best free coder now
REVIEW  = "nvidia/nemotron-mini-4b-instruct"
DEBUG   = "nvidia/nemotron-3-nano-30b-a3b"

# Single lightweight probe used for every model — short prompt, tiny output cap.
# Measures: latency, tokens/sec, and whether the model can follow a simple instruction.
BENCH_PROBE = {
    "system": "You are a coding assistant. Reply concisely.",
    "user":   "Write a Python one-liner that reverses a string. Reply with code only.",
    "max_tokens": 60,
    "score_fn": lambda r: (
        (50 if "[::-1]" in (r or "") else 0) +
        (30 if "def " in (r or "") or "lambda" in (r or "") else 0) +
        (20 if len((r or "").strip()) < 120 else 0)   # bonus for conciseness
    ),
}

# ----------------------------------------------------------------------

class TerminalAgent:
    def __init__(self):
        if not os.getenv("NVIDIA_API_KEY"):
            print("❌ NVIDIA_API_KEY environment variable not set!")
            print("   Export it:  export NVIDIA_API_KEY='nvapi-...'")
            sys.exit(1)

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )

    def _call(self, model, messages, max_tokens=1500, temp=0.2):
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    print(f"  ⚠️ API error: {e}")
                    return None
                time.sleep(2)

    def _call_timed(self, model, messages, max_tokens, temp=0.2):
        """Like _call but returns (content, latency_ms, tokens_out)."""
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                content = resp.choices[0].message.content or ""
                tokens_out = resp.usage.completion_tokens if resp.usage else len(content.split())
                return content, latency_ms, tokens_out
            except Exception as e:
                if attempt == 2:
                    return None, None, None
                time.sleep(2)

    # ------------------------------------------------------------------
    # Benchmark — all models fired in parallel, one tiny probe each
    # ------------------------------------------------------------------
    def benchmark(self, models=None, custom_task=None):
        """
        Fire all models simultaneously with a single short probe.
        Ranks by efficiency = quality / latency_seconds (higher is better).
        Typical wall-clock time: ~5-10 s regardless of model count.
        """
        models = models or ALL_MODELS
        probe  = BENCH_PROBE.copy()
        if custom_task:
            probe["user"] = custom_task

        msgs = [
            {"role": "system", "content": probe["system"]},
            {"role": "user",   "content": probe["user"]},
        ]

        print("\n" + "═" * 58)
        print("  🏁  EFFICIENCY BENCHMARK  (parallel, ~5-10 s)")
        print("═" * 58)
        print(f"  Probe: {probe['user'][:55]}")
        print(f"  Models: {len(models)}")
        print("═" * 58)

        def _probe(model):
            content, latency_ms, tokens_out = self._call_timed(
                model, msgs, max_tokens=probe["max_tokens"]
            )
            return model, content, latency_ms, tokens_out

        results = {}
        wall_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {pool.submit(_probe, m): m for m in models}
            done = 0
            for fut in as_completed(futures):
                model, content, latency_ms, tokens_out = fut.result()
                done += 1
                short = model.split("/")[-1]
                if content is None:
                    print(f"  [{done:2d}/{len(models)}] ❌  {short}")
                    results[model] = None
                    continue
                quality    = probe["score_fn"](content)
                efficiency = quality / (latency_ms / 1000) if latency_ms else 0
                tps        = round(tokens_out / (latency_ms / 1000), 1) if latency_ms else 0
                results[model] = {
                    "latency_ms": round(latency_ms),
                    "tokens_out": tokens_out,
                    "quality":    quality,
                    "efficiency": round(efficiency, 1),
                    "tps":        tps,
                }
                print(f"  [{done:2d}/{len(models)}] ✅  {short:<38}  "
                      f"{latency_ms:5.0f}ms  q={quality:3d}  {tps:5.1f} tok/s")

        wall_ms = (time.perf_counter() - wall_start) * 1000
        print(f"\n  ⏱  Total wall time: {wall_ms/1000:.1f} s\n")

        # Rank
        ranked = sorted(
            [(m, d) for m, d in results.items() if d],
            key=lambda x: x[1]["efficiency"],
            reverse=True,
        )

        print("═" * 58)
        print("  📊  RANKINGS  (quality / latency)")
        print(f"  {'#':<3} {'Model':<38} {'Eff':>6}  {'ms':>6}  {'tok/s':>6}")
        print(f"  {'-'*3} {'-'*38} {'------':>6}  {'------':>6}  {'------':>6}")
        medals = ["🥇", "🥈", "🥉"]
        for i, (model, d) in enumerate(ranked):
            prefix = medals[i] if i < 3 else f"  {i+1}."
            short  = model.split("/")[-1][:37]
            print(f"  {prefix} {short:<37} {d['efficiency']:>6.1f}  {d['latency_ms']:>6}ms  {d['tps']:>6.1f}")

        if ranked:
            winner = ranked[0][0]
            print(f"\n  🏆  Fastest & best: {winner}")
            print( "      Suggested config update:")
            print(f'      PLANNER = REVIEW = DEBUG = "{winner}"')
            if len(ranked) > 1:
                print(f'      CODER   = "{ranked[-1][0]}"  # most capable')

        print("═" * 58)

        out_path = WORKSPACE / "benchmark_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Results saved → {out_path}\n")
        return results

    # ------------------------------------------------------------------
    # Core agent capabilities (unchanged)
    # ------------------------------------------------------------------
    def plan(self, task):
        print("🧠 Planning...")
        msgs = [
            {"role": "system", "content": "Break this coding task into numbered steps. Keep steps concrete and small. Return only the numbered list."},
            {"role": "user", "content": task}
        ]
        resp = self._call(PLANNER, msgs, max_tokens=400)
        if not resp:
            return []
        steps = re.findall(r'\d+\.\s*(.+)', resp)
        if not steps:
            steps = [l.strip() for l in resp.splitlines() if l.strip()]
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        return steps

    def generate(self, instruction, context=""):
        print("💻 Generating code...")
        msgs = [
            {"role": "system", "content": (
                "You are a world-class programmer. Write clean, well-documented code. "
                "Return ONLY a JSON object with keys:\n"
                '  "files": {"filename": "full content"}, "explanation": "brief text"\n'
                "Do not wrap the JSON in markdown."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nTask: {instruction}"}
        ]
        resp = self._call(CODER, msgs, max_tokens=3000, temp=0.15)
        if not resp:
            return {}
        try:
            start = resp.find('{')
            end = resp.rfind('}') + 1
            return json.loads(resp[start:end])
        except:
            return {"files": {"code.py": resp}, "explanation": ""}

    def review(self, code, lang="python"):
        print("🔍 Reviewing...")
        msgs = [
            {"role": "system", "content": "Find bugs, style issues, and improvements. Be concise."},
            {"role": "user", "content": f"Review this {lang} code:\n```\n{code}\n```"}
        ]
        resp = self._call(REVIEW, msgs, max_tokens=800)
        return resp or "No review generated."

    def debug(self, code, error):
        print("🐞 Debugging...")
        msgs = [
            {"role": "system", "content": "Fix the code. Return ONLY the corrected code (no explanation)."},
            {"role": "user", "content": f"Code:\n```\n{code}\n```\nError:\n{error}"}
        ]
        resp = self._call(DEBUG, msgs, max_tokens=2000, temp=0.1)
        if resp:
            match = re.search(r'```(?:\w+)?\n(.*?)```', resp, re.DOTALL)
            if match:
                return match.group(1)
            return resp.strip()
        return code

    def explain(self, code):
        print("📖 Explaining...")
        msgs = [
            {"role": "system", "content": "Explain the code clearly but briefly."},
            {"role": "user", "content": f"Explain:\n```\n{code}\n```"}
        ]
        resp = self._call(PLANNER, msgs, max_tokens=600)
        return resp or ""

    def save_files(self, files, subdir=None):
        dir_path = WORKSPACE / (subdir or "")
        dir_path.mkdir(parents=True, exist_ok=True)
        saved = []
        for fname, content in files.items():
            fpath = dir_path / fname
            fpath.write_text(content)
            saved.append(str(fpath))
            print(f"  ✅ Saved: {fpath}")
        return saved

    def deploy_gstack(self, code, platform="cloudflare"):
        try:
            import httpx
            resp = httpx.post("http://127.0.0.1:8080/mcp/call", json={
                "tool": f"deploy_{platform}",
                "params": {"code": code}
            }, timeout=20)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

# ==============================================================
def interactive():
    agent = TerminalAgent()
    files_so_far = {}

    print("""
╔══════════════════════════════════════╗
║   🚀 NVIDIA TERMINAL CODER          ║
║   free models + agentic loop        ║
╚══════════════════════════════════════╝
Commands:
  /plan <task>          - break into steps
  /code <instruction>   - generate code
  /review               - review last generated code
  /debug <error>        - fix last code with given error
  /explain              - explain last code
  /save [name]          - save current files
  /files                - list saved files
  /models               - show available free models
  /bench                - run efficiency benchmark (all models)
  /bench fast           - benchmark only small/fast models
  /bench <model_name>   - benchmark a specific model
  /gstack deploy        - deploy via gstack (if running)
  /quit
""")

    print("📋 Active model config:")
    print(f"  ⚡ Planner    : {PLANNER}")
    print(f"  💻 Coder      : {CODER}")
    print(f"  🔍 Reviewer   : {REVIEW}")
    print(f"  🐞 Debugger   : {DEBUG}")
    print()

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not cmd:
            continue

        # ---- /models ----
        if cmd.startswith("/models"):
            print("📋 Active model config:")
            print(f"  ⚡ Planner    : {PLANNER}")
            print(f"  💻 Coder      : {CODER}")
            print(f"  🔍 Reviewer   : {REVIEW}")
            print(f"  🐞 Debugger   : {DEBUG}")
            print("\nAll available models:")
            for m in ALL_MODELS:
                print(f"  • {m}")
            continue

        # ---- /bench ----
        elif cmd.startswith("/bench"):
            arg = cmd[7:].strip().lower()
            if arg == "fast":
                models_to_test = ALL_MODELS[:5]   # only sub-8B models
            elif arg and arg != "fast":
                # user named a specific model or partial name
                matches = [m for m in ALL_MODELS if arg in m.lower()]
                models_to_test = matches or ALL_MODELS
                if not matches:
                    print(f"  ⚠️  No model matching '{arg}', running full benchmark.")
            else:
                models_to_test = ALL_MODELS
            agent.benchmark(models=models_to_test)

        # ---- /plan ----
        elif cmd.startswith("/plan"):
            task = cmd[6:].strip() or input("Task: ")
            agent.plan(task)

        # ---- /code ----
        elif cmd.startswith("/code"):
            instruction = cmd[6:].strip() or input("What to build? ")
            context = "\n".join(
                f"File: {n}\n```\n{c}\n```" for n, c in files_so_far.items()
            )
            result = agent.generate(instruction, context)
            if result.get("files"):
                files_so_far.update(result["files"])
                agent.save_files(result["files"])
                if result.get("explanation"):
                    print("  💡", result["explanation"])
            else:
                print("  ❌ Generation failed.")

        # ---- /review ----
        elif cmd.startswith("/review"):
            if not files_so_far:
                print("  No files in workspace. Generate code first.")
                continue
            all_code = "\n\n".join(f"// {n}\n{c}" for n, c in files_so_far.items())
            review = agent.review(all_code)
            print("📝 Review:\n" + review)

        # ---- /debug ----
        elif cmd.startswith("/debug"):
            error = cmd[7:].strip() or input("Error message: ")
            if not files_so_far:
                print("  No code to debug.")
                continue
            fname = max(files_so_far, key=lambda k: len(files_so_far[k]))
            code = files_so_far[fname]
            fixed = agent.debug(code, error)
            if fixed and fixed != code:
                files_so_far[fname] = fixed
                agent.save_files({fname: fixed})
                print("  🔧 Fixed and saved.")
            else:
                print("  No changes.")

        # ---- /explain ----
        elif cmd.startswith("/explain"):
            if not files_so_far:
                print("  No code to explain.")
                continue
            fname = max(files_so_far, key=lambda k: len(files_so_far[k]))
            explanation = agent.explain(files_so_far[fname])
            print("📖 Explanation:\n" + explanation)

        # ---- /save ----
        elif cmd.startswith("/save"):
            name = cmd[6:].strip() or "project"
            agent.save_files(files_so_far, subdir=name)

        # ---- /files ----
        elif cmd.startswith("/files"):
            print("📁 Workspace files:")
            for f in WORKSPACE.rglob("*"):
                if f.is_file():
                    print(f"  {f.relative_to(WORKSPACE)}")

        # ---- /gstack ----
        elif cmd.startswith("/gstack"):
            if not files_so_far:
                print("  No code to deploy.")
                continue
            code = "\n".join(files_so_far.values())
            result = agent.deploy_gstack(code)
            print("🚀 Deploy result:", json.dumps(result, indent=2))

        # ---- /quit ----
        elif cmd == "/quit":
            break

        else:
            print("🤖 Treating as full task...")
            steps = agent.plan(cmd)
            if steps:
                for step in steps:
                    input("\nPress Enter to run step...")
                    context = "\n".join(f"File: {n}\n```\n{c}\n```" for n, c in files_so_far.items())
                    res = agent.generate(step, context)
                    if res.get("files"):
                        files_so_far.update(res["files"])
                        agent.save_files(res["files"])

# ==============================================================
def one_off(args):
    agent = TerminalAgent()
    if args.benchmark is not None:
        # args.benchmark is a list; empty list = full bench, one item = filter/task
        custom_task = None
        models_to_test = ALL_MODELS
        if args.benchmark:
            val = args.benchmark[0].lower()
            if val == "fast":
                models_to_test = ALL_MODELS[:5]
            elif val.startswith("task:"):
                custom_task = val[5:].strip()
            else:
                matches = [m for m in ALL_MODELS if val in m.lower()]
                models_to_test = matches or ALL_MODELS
        if args.task:
            custom_task = args.task
        agent.benchmark(models=models_to_test, custom_task=custom_task)
    elif args.review:
        code = Path(args.review).read_text()
        print(agent.review(code))
    elif args.debug:
        code = Path(args.debug[0]).read_text()
        error = args.debug[1] if len(args.debug) > 1 else input("Error: ")
        print(agent.debug(code, error))
    elif args.explain:
        code = Path(args.explain).read_text()
        print(agent.explain(code))
    elif args.task:
        res = agent.generate(args.task)
        agent.save_files(res.get("files", {}))
    else:
        interactive()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      help="Direct task to generate code (also used as custom benchmark prompt)")
    parser.add_argument("--review",    help="Review a file")
    parser.add_argument("--debug",     nargs="+", help="Debug a file + error")
    parser.add_argument("--explain",   help="Explain a file")
    parser.add_argument(
        "--benchmark", nargs="*",
        metavar="[fast | MODEL_FILTER | task:PROMPT]",
        help=(
            "Run efficiency benchmark. Options:\n"
            "  (no arg)          test all models\n"
            "  fast              only sub-8B models\n"
            "  llama             filter models by name substring\n"
            "  task:'<prompt>'   use custom coding task as probe\n"
            "Combine with --task to inject a custom prompt."
        )
    )
    args = parser.parse_args()

    if any([args.benchmark is not None, args.task, args.review, args.debug, args.explain]):
        one_off(args)
    else:
        interactive()
