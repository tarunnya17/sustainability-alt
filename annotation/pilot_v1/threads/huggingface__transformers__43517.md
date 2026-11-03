# #43517 [pull_request] Make it possible to evaluate when using sequence parallel in HF Trainer
https://github.com/huggingface/transformers/pull/43517  ·  opened by jp1924 on 2026-01-27

## [body] jp1924 (CONTRIBUTOR) · opening post · 2026-01-27
# What does this PR do?

When using that sp(sequence parallel) in evaluate, the sequence length must be divisible by `world_size`, and the eval batch size must be the same as the train batch size to be able to use it, and this is a PR to modify that.
When using deepspeed sp, due to monkey patching the attention function, the following problems occur.

1. Eval batch size must be the same as train batch size.
2. During eval, the sequence must always be divisible by world_size.

In this situation, I propose to add the following two things in this PR.

1. A callback that restores the monkey-patched attention function to the original function during evaluate, and then reapplies it when training starts.
2. Check if it's training or not with self.model.training in the compute_loss section.

I think it should be made so that it can work without sequence parallel during evaluate.

However, some discussion is needed regarding these changes.
I'm not sure if when you first designed sequence parallel, you only considered it for training, or if you designed it to work together in evaluate as well.
Looking at the current code, it seems like it was made assuming it would be used in training, so I modified the code so that it can work normally in evaluate.

---

- `transformers` version: 5.0.0.dev0
- Platform: Linux-5.15.0-153-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Huggingface_hub version: 1.3.3
- Safetensors version: 0.7.0
- Accelerate version: 1.12.0
- Accelerate config:    not found
- DeepSpeed version: 0.18.4
- PyTorch version (accelerator?): 2.9.1+cu128 (CUDA)
- Using distributed or parallel set-up in script?: <fill in>
- Using GPU in script?: <fill in>
- GPU type: NVIDIA H100 NVL

I'm using the dev version via `pip install git+https://github.com/huggingface/transformers.git@dfe30827b8ebdd974eb7ce69c7d5d8cf8e6cf852`

## Before submitting

- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?

## Who can review?

@SunMarc

## [c2732702547] SunMarc (MEMBER) · review_comment on src/transformers/integrations/integration_utils.py:2434 · 2026-01-27
can you show me where the monkey patching due to sp happens ?

## [c3712029220] SunMarc (MEMBER) · review COMMENTED · 2026-01-27
Thanks for the PR, left a comment

## [c2732786530] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/integration_utils.py:2434 · reply to [c2732702547] · 2026-01-27
that is in the deepspeed i beleive: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L465-L466

## [c3712127943] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c3806181418] kashif (CONTRIBUTOR) · issue_comment · reply to [body] · 2026-01-27
good question @jp1924 lets me ask around if we want default, i.e. no SP for eval mode and then decide

## [c2732914504] jp1924 (CONTRIBUTOR) · review_comment on src/transformers/integrations/integration_utils.py:2434 · reply to [c2732702547] · 2026-01-27
@SunMarc 

The attention function changes occur through deepspeed's Accelerator's [accelerator.py > Accelerator > _prepare_deepspeed](https://github.com/huggingface/accelerate/blob/00e4d264f66d9694ba6116a0dbe7ecec83655b43/src/accelerate/accelerator.py#L2369-L2387) and then in deepspeed's [ulysses_sp.py > UlyssesSPAttentionHF > register_with_transformers](https://github.com/deepspeedai/DeepSpeed/blob/c89e0db8e2b061e9a1f872731b4d7bf92f628231/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L345-L468).

When checking the actual values in the vscode debug console, the results appear as shown in the image.

[image]


Looking at the image, you can see that it's divided into `_local_mapping` and `_global_mapping`, and you can confirm that `_local_mapping` has been changed to `UlyssesSPAttentionHF`.

## [c3712289140] jp1924 (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2732934090] jp1924 (CONTRIBUTOR) · review_comment on src/transformers/integrations/integration_utils.py:2434 · reply to [c2732702547] · 2026-01-27
I have some points I'd like to discuss:

1. Is it appropriate to add this to transformers?
   - I'm thinking it might be more appropriate to add this kind of functionality to accelerate instead

2. Is it right to handle this with callbacks?
   - Couldn't we make it automatically flow to global or local in `ALL_ATTENTION_FUNCTIONS` based on whether 
   model.training is True or False?
   - If there's a singleton-like approach (similar to Accelerator's PartialState) that's globally accessible without overhead and can globally check whether model.training is True or False, it seems like this could be easily applied here

What do you think?

## [c3712318280] jp1924 (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c3806347827] kashif (CONTRIBUTOR) · issue_comment · reply to [c3712318280] · 2026-01-27
@jp1924, another option might be to make the changes in deepspeed? we can make a context variable for eval that then when toggled skips the SP sharding in the deepspeed's uattn_wrapper ?

## [c3806446697] jp1924 (CONTRIBUTOR) · issue_comment · reply to [c3806347827] · 2026-01-27
Oh? You're right! Now that I think about it, modifying transformers or accelerate is just a temporary workaround, not a fundamental solution. @kashif your approach seems to be a more fundamental way to solve the problem.

## [c3851344834] jp1924 (CONTRIBUTOR) · issue_comment · reply to [c3806347827] · 2026-02-05
@kashif @SunMarc 
Great — the changes have been applied to DeepSpeed and Accelerate. 
Now it looks like we just need to apply the changes to transformers.

If writing tests is necessary, we might consider adding the test from this comment to transformers: https://github.com/deepspeedai/DeepSpeed/pull/7821#issuecomment-3822303457

## [c3853780688] SunMarc (MEMBER) · issue_comment · 2026-02-05
Yes, would you like to try writing a simple test to check that everything works ? You should try to come up with a test that doesn't take too much time to run !

## [c3860192755] kashif (CONTRIBUTOR) · issue_comment · reply to [c3853780688] · 2026-02-06
@SunMarc I added a change in the SP test to do eval on an odd sized input, but we will need to wait for a new version of deepspeed and accelerate before we can test this right?

## [c3860782042] SunMarc (MEMBER) · issue_comment · 2026-02-06
Yeah indeed ! But we can merge this even if it will fail for now.

## [c3763431882] SunMarc (MEMBER) · review APPROVED · 2026-02-06
Thanks a lot !

## [c3875368241] jp1924 (CONTRIBUTOR) · issue_comment · reply to [c3763431882] · 2026-02-10
@kashif @SunMarc 
First, I'm sorry for being late
I've been busy and couldn't check properly. But when I came back, 
the tests were already created? 

Thanks for helping! 
I'm sorry it seems like I left the work I should have done to you... ugh

## [c3877873791] SunMarc (MEMBER) · issue_comment · reply to [c3875368241] · 2026-02-10
> Thanks for helping!
> I'm sorry it seems like I left the work I should have done to you... ugh

It's @kashif who did it ! Thanks for your work and making the deepseed pr happen

## [c3882695553] jp1924 (CONTRIBUTOR) · issue_comment · 2026-02-11
thank you!

---
outcome: state=closed, merged=True
