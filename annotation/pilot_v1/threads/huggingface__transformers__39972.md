# #39972 [issue] Gemma3 with fp16 in inference (I don't know if this change is working in fine-tune) #BUG FIX
https://github.com/huggingface/transformers/issues/39972  ·  opened by DGTell on 2025-08-06

## [body] DGTell (NONE) · opening post · 2025-08-06
### First of all, I want to say that I’m not a programmer, and I don’t know much about GitHub.

(This post was translated with ChatGPT because my English isn’t great)

I don’t know if this issue is just with me or if it affects everyone, but **I managed to fix the problem** where, during inference with Gemma3-4B-it (I think it’s the same for other sizes), NaN values appeared in fp16.

### If Gemma3 works fine in fp16 for everyone reading this, then please just delete this post. Pleeeaseeee…

I just want to share some thoughts. If you find this useless, just scroll to the end :3

- [My personal experience and opinion](#My-personal-experience-and-opinion)
- [Where the overflow happens](#Where-the-overflow-happens)
- [The fix](#The-fix)
- [That’s it](#Thats-it)

# My personal experience and opinion

My hardware isn’t very powerful, so fp16 was my only hope. But with this model, CUDA kernel crashed almost immediately. I thought, “It’s 100% vision-tower’s fault,” and tried loading those layers in bf16 (simulated on Windows 11) while keeping the rest in fp16, but as I said, I’ve only been using Python for less than a month – so I failed.

After another failed attempt to load the model in fp16, I got frustrated and dove into GitHub forums (and here), and Hugging Face, but since I’m not good at searching, I couldn’t find posts about this specific problem. I did see an inspiring post where someone (or some team) fixed a similar issue for a diffusion model.

That’s when I realized: I need to get into the transformers/pytorch code and fix it myself.

I went into the file that threw the CUDA kernel error `.../generation/utils.py`, and for the first time saw that the crash was caused by NaN. Together with ChatGPT, I tried to find the root cause, but after a whole day of debugging, we found nothing.

A few days later, I came back to the problem, and I don’t remember how, but I ended up in `.../models/gemma3/modeling_gemma3.py` (Yeah, I know it’s auto-generated, but for quick fixes, it’s perfect).

So, I figured out: CUDA doesn’t crash right away, but after 5–10 seconds. That means there’s an **overflow** (or blowup) of values. Want to know my debugging process? ChatGPT told me the most likely spots for float16 overflow, and I just spammed `print('Variable name:', variable.max().item())` INSIDE EVERY def forward. My console wanted to cry. But I was happy as a kid, because with every printout ChatGPT pointed out what looked suspicious. He even wrote a function to detect exactly where Inf/NaN first appeared during a forward. At one point, my script had around 130 print statements just for debugging.

And I found exactly where the NaN (and before that, inf) appeared! Finally, here’s something useful for you:

# Where the overflow happens

The reason is in the following lines (I don’t know how to make a diff format in markdown to highlight lines, sorry):

## First cause:

```python
132 class Gemma3RMSNorm(nn.Module):
133    def __init__(self, dim: int, eps: float = 1e-6):
134        super().__init__()
135        self.eps = eps
[… 11 lines of code omitted]

## Second cause:

```python
373        residual = hidden_states
374
375        hidden_states = self.input_layernorm(hidden_states)
376
[… 25 lines of code omitted]

The first cause may not be obvious, but for some reason, the model weights there keep changing. ChatGPT said this shouldn’t happen during inference (with my inference code, it’s fine). So I added to my code:

```python
for param in model.parameters():
    param.requires_grad = False
```

And this seemed to fix the problem—no more overflows in `return output.type_as(x)`, though I’m not 100% sure because I forgot what print was showing after my fix. Yeah, since we’re returning to fp16, it immediately gives inf (something in gemma3 code changes weights incorrectly, as `print(self.weight.max().item())` kept showing different and increasing values every forward).

The second cause became obvious when I saw this:

```python
373        residual = hidden_states
...
395        hidden_states = residual + hidden_states
396
397        residual = hidden_states
...
401        hidden_states = residual + hidden_states
```

This basically means that with each forward pass, the values keep increasing—and sooner or later, even fp64 will overflow.

# The fix:

## It’s very simple:

```python
142        output = self._norm(x.float())
143        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
144        # See https://github.com/huggingface/transformers/pull/29402
145        output = output * (1.0 + self.weight.float())
146+       fp16_max = 65503.9
147+       output = torch.clamp(output, min=-fp16_max, max=fp16_max)
148        return output.type_as(x)
```

## And…

```python
397+       hidden_states = (residual + hidden_states).float()
398+       fp16_max = 65503.9
399+       hidden_states = torch.clamp(hidden_states, min=-fp16_max, max=fp16_max)
400+       hidden_states = hidden_states.half()
401
402        residual = hidden_states
403        hidden_states = self.pre_feedforward_layernorm(hidden_states)
404        hidden_states = self.mlp(hidden_states)
405        hidden_states = self.post_feedforward_layernorm(hidden_states)
406+       hidden_states = (residual + hidden_states).float()
407+       hidden_states = torch.clamp(hidden_states, min=-fp16_max, max=fp16_max)
408+       hidden_states = hidden_states.half()
```

That’s it—fp16 works for me now! And, for some reason, it works even better than the original fp32/bf16 (sometimes the model used to generate gibberish for no reason). But I want to be clear:

**THIS CODE IS NOT IDEAL** Yes, it solves the NaN problem, but for some reason, inf still appears in hidden_states, but then disappears, and when the model generates tokens, everything is fine. But the fact that in the original code, hidden_states in fp32 becomes ~264,000.0 is NOT NORMAL. Nor is it normal for model weights to change unless include this in inference code:

```python
for param in model.parameters():
    param.requires_grad = False
```

Even though I tried adding `model = Gemma3ForConditionalGeneration.from_pretrained(...).eval()` and `with torch.inference_mode():` And the example from Gemma3’s Hugging Face card has the same issues, so it’s NOT a problem with my inference code!

## That’s it

I’m not going to submit a PR or edit the repo for two reasons:

1. This is a hack; it shouldn’t be necessary in the first place. (According to ChatGPT, weights should not change during inference. As for hidden_states... I don’t think they should behave this way either…)
    
2. It could be fixed better (like `if dtype == fp16:` or something—BUT I’M NOT A PROGRAMMER!!!)

## [c3265810768] Rocketknight1 (MEMBER) · issue_comment · 2025-09-08
This was a fun read! The fundamental problem isn't really a bug, though, it's that **Gemma3 was never intended to be run in fp16.** Some models, by design, operate in value ranges beyond the fp16 range - recent versions of `Pixtral` are a good example of this. The vision tower in those models just outputs values >= `1e5` and these often overflow `fp16`. There's not much you can do about this - a clamp can temporarily fix it, but it will reduce model quality.

Ideally, you should run these models in `bfloat16`, which is how they were trained. If your hardware is too old to support `bfloat16`, then you should consider doing inference in another framework. For example, you can use `llama-cpp` or `llama-cpp-python` and load an 8-bit GGUF file. This will use even less memory than `bfloat16` and should work on older hardware too. Quality should be very close to the original. The downside is you cannot use these formats for training, but for inference it might be ideal.

---
outcome: state=closed
