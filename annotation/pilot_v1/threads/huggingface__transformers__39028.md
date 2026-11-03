# #39028 [pull_request] Granite speech speedup + model saving bugfix
https://github.com/huggingface/transformers/pull/39028  ·  opened by avihu111 on 2025-06-25

## [body] avihu111 (CONTRIBUTOR) · opening post · 2025-06-25
# What does this PR do?

## Speeding up the encoder
[Reverting](https://github.com/huggingface/transformers/pull/36801#discussion_r2013681886) Shaw's positional embedding calculation to einsum results in a significant speedup in both inference/training runtime. 
We found it to be x30 times faster than the current explicit dot product using bfloat16.
I kept the explicit dot product in a comment for readability.
I hope that it would be possible.

## Fixing issues with loading and saving with an adapter

- When saving a checkpoint, the adapter config pointed to the original model, instead of the updated model
- It fixes a bug,  where we changed `_hf_peft_config_loaded` when saving
- It reverts a tensor renaming that was triggered by adding an adapter.

Maybe there's a better solution for the problems I was facing - I'll be happy to hear your opinion.
I added comments on each code change, along with the necessary context and justification for the change. 

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?
@ArthurZucker  @eustlb can you give that a look? 🙏 
CC: @avishaiElmakies @alex-jw-brooks

## [c2166025211] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:166 · 2025-06-25
`einsum` runs significantly faster (measured with 500 repetitions), and has a smaller memory footprint:
```
einsum: 25.089 ms
existing (explicit dot): 594.220 ms
``` 
I was hoping we could use the einsum implementation to speed up inference/finetuning, and keep the equivalent formulation for readability.

## [c2957015091] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-25 (no text)

## [c2166027942] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:552 · 2025-06-25
ensures the adapter config points to the finetuned model

## [c2957019413] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-25 (no text)

## [c2166029390] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:558 · 2025-06-25
bugfix, ensuring `save_pretrained` would not change the original value

## [c2957021651] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-25 (no text)

## [c2166036545] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:562 · 2025-06-25
The adapter changes the original parameter names by adding: `.base_layer` to each one.
This hack enables `save_pretrained()` and `from_pretrained()` to work as expected.

## [c2957032694] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-25 (no text)

## [c2166727191] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:166 · 2025-06-25
If einsum is possible I am fairly sur there is a way to do this with just matrix notation! We always avoid einsum in transformers! 
Let's add a TODO here as you probably want to have this merged fast!

## [c2958172864] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-06-25
thanks! happy to merge as is but in general einsum is not magic, there is an equivalent implementation out there that only uses matrix notation!

## [c2166957227] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:166 · reply to [c2166727191] · 2025-06-25
I really tried finding an alternative!
`matmul` is not a great choice, since we need vectorized dot product, and matmul would have redundant computations. 
[vecdot](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vecdot.html) was a promising direction, but it was actually slower. 
I think the einsum speedup has to do with either broadcasting, kernels for half precision - not sure. 
I'll make sure to update it if I'll learn something new. 
Adding a todo - Thanks Arthur!

## [c2958568961] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-25 (no text)

## [c2167052268] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:166 · reply to [c2166727191] · 2025-06-25
Hi @avihu111, just saw this while on watch! Try

```
(query_states.unsqueeze(-2) @ rel_pos_emb.transpose(-1, -2)).squeeze(-2)
```

## [c2958728443] Rocketknight1 (MEMBER) · review COMMENTED · 2025-06-25 (no text)

## [c2168186730] avihu111 (CONTRIBUTOR) · review_comment on src/transformers/models/granite_speech/modeling_granite_speech.py:166 · reply to [c2166727191] · 2025-06-26
Thanks @Rocketknight1 for a great suggestion. 
That's a cool trick to get `bmm` to perform vectorized dot product. 
For some reason, it still performs on par with the explicit dot product, which is still x50 times slower than einsum 😮 

I ran the following code to compare all methods:
```
        for method in ["einsum", "explicit_dot", "vecdot", "bmm"]:
            with torch.amp.autocast("cuda", torch.bfloat16):
                t1 = time.time()
                for _ in range(500):
                    if method == "einsum":
[… 19 lines of code omitted]
```

## [c2960682097] avihu111 (CONTRIBUTOR) · review COMMENTED · 2025-06-26 (no text)

## [c3007518689] ArthurZucker (COLLABORATOR) · issue_comment · 2025-06-26
Thanks for the thorough checks ! 🤗 makes a lot of sense when we have this huge perf diff!

---
outcome: state=closed, merged=True
