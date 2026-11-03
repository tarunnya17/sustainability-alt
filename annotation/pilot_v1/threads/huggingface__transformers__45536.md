# #45536 [issue] Feature Request: Add SCAO Optimizer integration for 1.5x faster fine-tuning throughput
https://github.com/huggingface/transformers/issues/45536  ·  opened by whispering3 on 2026-04-20

## [body] whispering3 (NONE) · opening post · 2026-04-20
### Feature request

Currently, AdamW is the default standard for fine-tuning via the Trainer class. While robust, its diagonal approximation of loss curvature makes early convergence slow, which is particularly expensive in compute-constrained environments or rapid fine-tuning pipelines (like LoRA/PEFT) where wall-clock time is the primary bottleneck.

I have developed and open-sourced SCAO (Sparse Curvature-Aware Adaptive Optimizer), a second-order PyTorch optimizer designed as a drop-in replacement for AdamW. It delivers Shampoo-quality preconditioned gradients but uses adaptive rank selection (keeping only top-k eigenvectors) to make it highly efficient.

I would like to propose integrating SCAO as a supported optimizer within TrainingArguments (e.g., optim="scao").

Experimental Results & Benchmarks
In our benchmarks against AdamW, SCAO demonstrates a significant advantage in both raw throughput and scaling capabilities:

1M Parameter Model (500 steps):

Throughput: 827 tok/s vs AdamW's 537 tok/s (+54% faster).

Time-to-Target: Reaches a PPL of 14.10 in just 320 seconds (AdamW takes 582s).

Quality: Final PPL gap is negligible (+0.18 PPL compared to AdamW), but achieved in ~60% of the wall-clock time.

5M Parameter Model (Scaling Dominance):

As model complexity increases, SCAO's second-order approximation shines. At 5M parameters, SCAO beat AdamW by 2.55 PPL (23.94 vs 26.49), an improvement of ~9.6%.

Note: SCAO introduces a ~2.2x memory overhead compared to AdamW due to the sparse preconditioner matrices, which dilutes at larger batch sizes. It implements a fallback diagonal approximation for layers exceeding max_precond_dim to prevent OOM errors.

Proposed Implementation Design
Since SCAO inherits from torch.optim.Optimizer, the integration is straightforward:

Add "scao" to OptimizerNames in training_args.py.

Add the initialization logic in Trainer.create_optimizer().

Allow passing SCAO-specific kwargs (like precond_freq, warmup_steps) via optim_args.

Code repository and Paper

GitHub Repo: https://github.com/whispering3/scao

Next Steps
I already have a working integration of SCAO with the Trainer class locally. If the core team is open to this addition, I would be more than happy to submit a Pull Request with the implementation, comprehensive tests, and documentation.

Would love to hear the maintainers' thoughts on this!

### Motivation

I'm always frustrated when fine-tuning LLMs in compute-constrained environments because AdamW's diagonal approximation of loss curvature makes early convergence slow. Wall-clock time is a primary bottleneck for rapid fine-tuning pipelines (like LoRA/PEFT).

### Your contribution

I already have a working integration of SCAO with the Trainer class locally, and I am ready to submit a PR if the core team is open to this addition.

Since SCAO inherits from torch.optim.Optimizer, the proposed implementation is very straightforward:

Add "scao" to OptimizerNames in training_args.py.
Add the initialization logic in Trainer.create_optimizer().
Allow passing SCAO-specific kwargs via optim_args.
Let me know if I should go ahead and open the Pull Request!

## [c4288410855] Rocketknight1 (MEMBER) · issue_comment · 2026-04-21
Probably not, I'm sorry! We generally don't like adding very new/hot optimizers until we see some real-world evidence that they're being adopted. Also, that whole Github repo is clearly written by LLM, so I'm not convinced the whole thing isn't semi-hallucinated, and I'd basically have to rerun all your experiments to verify it, which I don't have time to do! 😅

---
outcome: state=closed
