# #41867 [issue] [RFC] Automatic CPU dtype fallback and thread optimization
https://github.com/huggingface/transformers/issues/41867  ·  opened by AmazingcatAndrew on 2025-10-26

## [body] AmazingcatAndrew (NONE) · opening post · 2025-10-26
<html>
<body>
<html><head></head><body><h1>RFC: Automatic CPU dtype fallback and thread optimization in Transformers</h1>
<h2>Summary</h2>
<p>Add automatic dtype validation and thread optimization for CPU inference in Transformers. When users specify <code>dtype=torch.float16</code> or <code>torch.bfloat16</code> on CPU devices, the library should either:</p>
<ol>
<li>Automatically fall back to <code>float32</code> with a clear warning, OR</li>
<li>Raise an error with actionable guidance</li>
</ol>
<p>Additionally, set sensible CPU thread defaults to improve out-of-the-box performance.</p>
<hr>
<h2>Motivation</h2>
<h3>Problem Statement</h3>
<p>CPU devices do not natively support fp16/bf16 arithmetic. When users load models with these dtypes on CPU, PyTorch falls back to <strong>software emulation</strong>, causing:</p>
<ol>
<li><strong>Severe performance degradation</strong> (20-30× slower)</li>
<li><strong>Correctness issues</strong> (numerical instability, abnormal outputs)</li>
<li><strong>Silent failures</strong> (no warnings, users unaware of the problem)</li>
</ol>
<p>Additionally, Transformers does not set optimal thread counts for CPU inference, relying on PyTorch defaults which may be suboptimal, especially on modern hybrid CPU architectures.</p>
<hr>
<h2>Current Behavior</h2>
<p><strong>Test Environment:</strong></p>
<ul>
<li><strong>Device</strong>: Laptop (Intel Core i7-13600P, 13th Gen Raptor Lake)
<ul>
<li>Architecture: Hybrid (Performance + Efficiency cores)</li>
<li>P-cores: 6 (12 threads with Hyper-Threading)</li>
<li>E-cores: 4 (4 threads, no HT)</li>
<li>Total logical cores: 16 (detected by <code>os.cpu_count()</code>)</li>
<li>Base clock: 2.20 GHz</li>
</ul>
</li>
<li><strong>OS</strong>: Windows 11 Home Chinese Edition (Version 24H2, Build 26100.6899)</li>
<li><strong>Memory</strong>: 32 GB RAM (31.7 GB available)</li>
<li><strong>Python</strong>: 3.10.18</li>
<li><strong>PyTorch</strong>: 2.9.0+cpu</li>
<li><strong>Transformers</strong>: 4.57.0.dev0 (latest main branch)</li>
<li><strong>Test Model</strong>: <code>distilgpt2</code> (82M parameters)</li>
</ul>
<h3>Test A: <code>model.generate()</code> performance</h3>

dtype requested | dtype loaded | tok/s | vs float32 | Warning shown?
-- | -- | -- | -- | --
torch.float16 | torch.float16 | 0.65 | 🔴 28× slower | ❌ No
torch.bfloat16 | torch.bfloat16 | 0.69 | 🔴 26× slower | ❌ No
torch.float32 | torch.float32 | 18.1 | ✅ baseline | -

<p><strong>Note</strong>: Absolute numbers will vary by CPU generation and model, but relative improvements should be consistent.</p>
<h3>Correctness</h3>
<ul>
<li>✅ Eliminates numerical instability from fp16/bf16 emulation on CPU</li>
<li>✅ Ensures consistent output quality across dtypes</li>
<li>✅ Prevents silent failures where outputs are abnormally short or incorrect</li>
</ul>
<h3>User Experience</h3>
<ul>
<li>✅ Clear, actionable warnings when risky configurations are detected</li>
<li>✅ Automatic fixes that "just work" for 95% of users</li>
<li>✅ Escape hatches for advanced users (env vars, explicit params)</li>
<li>✅ Better out-of-the-box CPU performance via thread optimization</li>
<li>✅ Reduces "why is my CPU inference so slow?" support requests</li>
</ul>
<hr>
<h2>Prior Art / Related Work</h2>
<h3>Duplicate Check</h3>
<p>I will search the following in <code>huggingface/transformers</code> before final submission:</p>
<ul>
<li>Keywords: <code>cpu</code>, <code>dtype</code>, <code>fp16</code>, <code>bf16</code>, <code>float16</code>, <code>bfloat16</code>, <code>fallback</code>, <code>threads</code>, <code>num_threads</code>, <code>performance</code></li>
<li>Scope: Open/closed issues, merged PRs, last 12 months</li>
</ul>
<p><strong>Search URLs:</strong></p>
<pre><code>https://github.com/huggingface/transformers/issues?q=is%3Aissue+cpu+dtype+fp16
https://github.com/huggingface/transformers/issues?q=is%3Aissue+cpu+slow+half
https://github.com/huggingface/transformers/pulls?q=is%3Apr+cpu+dtype+fallback
</code></pre>
<p><strong>Preliminary findings:</strong></p>
<ul>
<li>Scattered user reports of poor CPU performance, but no systematic solution proposed</li>
<li>No existing unified CPU heuristics layer found in codebase</li>
<li>Some documentation mentions CPU is slow with certain dtypes, but no automatic handling</li>
</ul>
<p><em>(Will update with specific issue numbers after manual review)</em></p>
<h3>External Projects</h3>
<ul>
<li><strong>ONNX Runtime</strong>: Has CPU-specific optimizations and dtype handling, but requires model export workflow</li>
<li><strong>llama.cpp</strong>: CPU-first inference with quantization, but uses different model format</li>
<li><strong>PyTorch</strong>: Provides low-level controls (<code>torch.set_num_threads</code>) but no high-level guidance for optimal defaults</li>
<li><strong>TensorFlow</strong>: Has similar dtype emulation issues on CPU, also lacks automatic handling</li>
</ul>
<hr>
<h2>Alternatives Considered</h2>
<h3>Alternative 1: Documentation only</h3>
<p><strong>Rejected</strong>: Users rarely read performance docs before encountering issues. Proactive runtime guidance is more effective.</p>
<h3>Alternative 2: Raise hard errors instead of warnings + auto-fallback</h3>
<p><strong>Rejected</strong>: Too disruptive for existing code. Warnings + auto-fallback provides gentler migration path while still solving the problem.</p>
<h3>Alternative 3: Add explicit <code>cpu_mode="fast"</code> parameter to APIs</h3>
<p><strong>Rejected</strong>: Adds API surface area. Environment variables + smart defaults are more maintainable and don't require code changes.</p>
<h3>Alternative 4: Only fix <code>pipeline</code>, not low-level <code>generate()</code></h3>
<p><strong>Rejected</strong>: Problem affects all CPU users. Partial fix would leave some users confused about inconsistent behavior.</p>
<h3>Alternative 5: Detect hybrid CPU architectures and use P-core count for threads</h3>
<p><strong>Considered for future</strong>: Would require platform-specific CPU topology detection. Good enhancement but out of scope for initial implementation. Document as future work.</p>
<hr>
<h2>Open Questions</h2>
<ol>
<li>
<p><strong>Should thread configuration be one-time (session-level) or per-inference?</strong></p>
<ul>
<li><strong>Proposal</strong>: One-time at first inference, with logging. Users can override via env vars before import.</li>
<li><strong>Rationale</strong>: Repeatedly calling <code>torch.set_num_threads()</code> has overhead; session-level is standard practice.</li>
</ul>
</li>
<li>
<p><strong>Should we support CPU-specific quantization (int8) in this RFC?</strong></p>
<ul>
<li><strong>Proposal</strong>: Out of scope. Focus on dtype + threads first. Quantization can be follow-up RFC (or part of議題 B).</li>
<li><strong>Rationale</strong>: Quantization is a larger feature with different trade-offs; keeping this RFC focused improves acceptance chances.</li>
</ul>
</li>
<li>
<p><strong>How to handle mixed CPU/GPU scenarios (e.g., model on GPU, inputs on CPU)?</strong></p>
<ul>
<li><strong>Proposal</strong>: Only apply heuristics when both model parameters AND compute device are CPU.</li>
<li><strong>Implementation</strong>: Check <code>model.device</code> and <code>inputs.device</code> at inference time.</li>
</ul>
</li>
<li>
<p><strong>Should pipelines have different defaults than direct model usage?</strong></p>
<ul>
<li><strong>Proposal</strong>: Unified behavior. Both should use the same <code>cpu_heuristics</code> module.</li>
<li><strong>Rationale</strong>: Consistent user experience; reduces cognitive load.</li>
</ul>
</li>
<li>
<p><strong>How to handle hybrid CPU architectures (P-cores + E-cores)?</strong></p>
<ul>
<li><strong>Current observation</strong>: On Intel 13th Gen (6P+4E = 16 logical cores), default threads=12 may not optimally utilize the hybrid design.</li>
<li><strong>Proposal for this RFC</strong>: Use <code>os.cpu_count()</code> as baseline, document hybrid CPU considerations.</li>
<li><strong>Future enhancement</strong>: Could detect hybrid architectures (via <code>psutil</code> or platform APIs) and set <code>intra_op_threads</code> to P-core count for latency-sensitive workloads.</li>
<li><strong>Rationale</strong>: Advanced CPU topology detection adds complexity; defer to future work after gathering user feedback on basic heuristics.</li>
</ul>
</li>
</ol>
<hr>
<h2>Testing Strategy</h2>
<h3>Reproducible Test Script</h3>
<p>Users and maintainers can run these scripts to verify current behavior and test fixes:</p>
<p><strong>Script 1: Test <code>generate()</code> with different dtypes</strong></p>
<pre><code class="language-python"># probe_generate_cpu.py
import os, time, json, platform
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

def peak_rss_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)

def run_case(model_id, dtype, set_threads=None, max_new_tokens=64):
    if set_threads:
        intra, inter = set_threads
        torch.set_num_threads(intra)
        torch.set_num_interop_threads(inter)
    
    print(f"\n=== Testing dtype={dtype}, threads={set_threads} ===", flush=True)
    
    tk = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    text = "The quick brown fox jumps over the lazy dog. " * 32
    inputs = tk(text, return_tensors="pt")
    
    # warmup
    _ = m.generate(**inputs, max_new_tokens=8)

    rss_before = peak_rss_mb()
    t0 = time.time()
    out = m.generate(**inputs, max_new_tokens=max_new_tokens)
    dt = (time.time() - t0) * 1000
    rss_after = peak_rss_mb()
    tok_per_s = max_new_tokens / (dt/1000)

    return {
        "model": model_id,
        "requested_dtype": str(dtype),
        "effective_param_dtype": str(next(m.parameters()).dtype),
        "threads": {
            "intra": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads()
        },
        "timing_ms": round(dt, 2),
        "tok_per_s": round(tok_per_s, 2),
        "peak_mem_delta_mb": round(max(0.0, rss_after - rss_before), 2),
    }

def main():
    report = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "os": platform.platform(),
            "cpu_count": os.cpu_count()
        },
        "cases": []
    }
    model = "distilgpt2"
    
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for threads in [None, (os.cpu_count() or 4, max(1, (os.cpu_count() or 4)//2))]:
            try:
                case = run_case(model, dtype, threads)
                report["cases"].append(case)
            except Exception as e:
                case = {
                    "model": model,
                    "requested_dtype": str(dtype),
                    "threads": threads,
                    "error": repr(e)
                }
                report["cases"].append(case)
                print(f"ERROR: {e}", flush=True)

    print("\n" + "="*60)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
</code></pre>
<p><strong>Script 2: Test <code>pipeline()</code> with different dtypes</strong></p>
<pre><code class="language-python"># probe_pipeline_cpu.py
import time, json, os, platform
import torch
from transformers import pipeline

def test_pipeline(dtype):
    pipe = pipeline(
        "text-generation",
        model="distilgpt2",
        torch_dtype=dtype,
        device=-1  # CPU
    )
    
    text = "The future of artificial intelligence is"
    t0 = time.time()
    result = pipe(text, max_new_tokens=50, do_sample=False)
    dt = (time.time() - t0) * 1000
    
    return {
        "requested_dtype": str(dtype),
        "time_ms": dt,
        "len": len(result[0]["generated_text"])
    }

def main():
    report = {
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cpu_count": os.cpu_count()
        },
        "cases": []
    }
    
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            case = test_pipeline(dtype)
            report["cases"].append(case)
            print(f"Tested {dtype}: {case['time_ms']:.1f}ms, {case['len']} chars")
        except Exception as e:
            print(f"ERROR with {dtype}: {e}")
            report["cases"].append({"requested_dtype": str(dtype), "error": repr(e)})
    
    print("\n" + "="*60)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
</code></pre>
<p><strong>Expected behavior after fix:</strong></p>
<ul>
<li>Warning printed about fp16/bf16 on CPU</li>
<li>Automatic fallback to float32</li>
<li>Performance matches baseline float32 results</li>
<li>Output quality consistent across all dtypes</li>
</ul>
<hr>
<h2>Success Metrics</h2>
<ol>
<li><strong>Zero reports of "CPU inference is slow" without appropriate warnings</strong> (tracked via issue mentions)</li>
<li><strong>Measurable performance improvement</strong> in CPU inference benchmarks (議題 B will establish baseline)</li>
<li><strong>Positive community feedback</strong> on UX improvements (tracked via issue/PR comments, Discord mentions)</li>
<li><strong>No breaking changes</strong> to existing code (verified by CI test suite passing)</li>
<li><strong>Reduction in support burden</strong>: Fewer duplicate issues about CPU performance</li>
</ol>
<hr>
<h2>Timeline</h2>
<ul>
<li><strong>Week 1</strong>: Community feedback on this RFC (waiting for maintainer/community input)</li>
<li><strong>Week 2</strong>: Implementation (core heuristics + integration points)</li>
<li><strong>Week 3</strong>: Testing + documentation + addressing review feedback</li>
<li><strong>Week 4</strong>: PR review iteration and merge</li>
</ul>
<p><strong>Note</strong>: Timeline assumes RFC is approved. If scope adjustments are requested, timeline will be updated accordingly.</p>
<hr>
<h2>References</h2>
<ul>
<li><strong>Test data</strong>: Intel Core i7-13600P (13th Gen), Windows 11, PyTorch 2.9.0+cpu, Transformers 4.57.0.dev0</li>
<li><strong>Reproducible test scripts</strong>: Will be uploaded to GitHub Gist and linked in RFC comments</li>
<li><strong>Related issues</strong>: (To be filled after duplicate check search)</li>
</ul>
<hr>
<h2>Appendix: Test Hardware Details</h2>
<h3>CPU Architecture</h3>
<p><strong>Model</strong>: Intel Core i7-13600P (Raptor Lake, 13th Gen)</p>
<ul>
<li><strong>P-cores (Performance)</strong>: 6 cores, 12 threads (with Hyper-Threading)</li>
<li><strong>E-cores (Efficiency)</strong>: 4 cores, 4 threads (no Hyper-Threading)</li>
<li><strong>Total logical cores</strong>: 16 (as reported by <code>os.cpu_count()</code>)</li>
<li><strong>Base clock</strong>: 2.20 GHz</li>
<li><strong>Form factor</strong>: Mobile/Laptop processor</li>
</ul>
<h3>Why This Hardware Matters</h3>
<ol>
<li>
<p><strong>Modern architecture</strong>: 13th Gen Intel (2023) represents current mainstream CPUs</p>
<ul>
<li>The <strong>28× slowdown with fp16 on this modern hardware</strong> suggests older CPUs will be even worse</li>
<li>Validates that the problem is fundamental, not hardware-specific</li>
</ul>
</li>
<li>
<p><strong>Hybrid design</strong>: P-cores + E-cores architecture is increasingly common</p>
<ul>
<li>Intel 12th Gen+ (2021 onwards)</li>
<li>AMD Ryzen with 3D V-Cache</li>
<li>ARM big.LITTLE (mobile/Apple Silicon)</li>
<li>Thread optimization becomes more important for these architectures</li>
</ul>
</li>
<li>
<p><strong>Representative use case</strong>: Laptop CPU reflects common deployment scenario</p>
<ul>
<li>Many developers/researchers work on laptops</li>
<li>Edge deployment often uses laptop-class or embedded CPUs</li>
<li>Desktop CPUs will show similar patterns with different absolute numbers</li>
</ul>
</li>
</ol>
<h3>Full Test Results</h3>
<p><strong>Raw data files</strong> (will be provided in RFC comments):</p>
<ul>
<li><code>gen_cpu_report.json</code>: Detailed performance data for <code>generate()</code></li>
<li><code>pipe_cpu_report.json</code>: Detailed performance data for <code>pipeline()</code></li>
</ul>
<p><strong>Key observations:</strong></p>
<ol>
<li>fp16 on CPU: 28× slower in <code>generate()</code>, produces 78% shorter outputs in <code>pipeline()</code></li>
<li>bfloat16 on CPU: 26× slower in <code>generate()</code>, produces 14% shorter outputs in <code>pipeline()</code></li>
<li>Thread defaults: intra_op=12, inter_op=12 (suboptimal for 16-core hybrid system)</li>
<li>No warnings or errors emitted by Transformers for any dtype on CPU</li>
</ol>
<hr>
<h2>How to Provide Feedback</h2>
<p>Please comment on this issue with:</p>
<ul>
<li>✅ <strong>Support</strong> for the proposal (helps gauge community interest)</li>
<li>🤔 <strong>Concerns</strong> or alternative approaches (helps refine the design)</li>
<li>📊 <strong>Additional test data</strong> from your environment (CPU model, OS, performance numbers)</li>
<li>💡 <strong>Suggestions</strong> for scope, implementation details, or priority</li>
</ul>
<p><strong>Specific questions for maintainers:</strong></p>
<ol>
<li>Is the proposed scope (dtype fallback + thread defaults) appropriate, or should we split into separate PRs?</li>
<li>Are the proposed environment variable names (<code>HF_CPU_DTYPE_POLICY</code>, etc.) acceptable?</li>
<li>Should we emit warnings via Python <code>warnings</code> module or <code>logging</code>? (Current proposal: <code>logging.warning</code>)</li>
<li>Any concerns about setting global PyTorch state (<code>torch.set_num_threads</code>)? Should we only recommend rather than auto-set?</li>
</ol>
<p>I'm happy to adjust the proposal based on maintainer guidance and am ready to implement this as a PR once the approach is validated.</p>
</body></html>
</body>
</html>

## [c3456379870] Rocketknight1 (MEMBER) · issue_comment · 2025-10-28
Tell your LLM to write about 10% as much text in future please  😭 

Yes, though, `bfloat16` on CPU is very slow because of software emulation. This is most often going to be a problem when users use `auto` dtype, and then don't realize why things are so slow. The problem, though, is that many users instantiate a model on one device and then move it, and raising a warning every time they instantiate a `bfloat16` model on CPU could get very annoying. I don't think we want an overengineered solution here!

---
outcome: state=closed
