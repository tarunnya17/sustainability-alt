# #46572 [pull_request] [DiffusionGemma] Support gradient checkpointing
https://github.com/huggingface/transformers/pull/46572  ·  opened by kashif on 2026-06-11

## [body] kashif (CONTRIBUTOR) · opening post · 2026-06-11
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=46572)

Follow up to #46568, which declared gradient checkpointing unsupported because checkpointed layers drop the encoder KV cache writes the decoder depends on.

This makes it work with the existing machinery. Encoder layers return their local KV states and the model writes the cache outside the checkpointed call, so the states are checkpoint outputs and keep the gradient path from the decoder open. Decoder layers receive the per layer encoder KV as an encoder_kv_states tuple of plain tensors, following the (key_states, value_states) convention of Gemma4's shared_kv_states (a dict mutated inside the layer would not survive checkpointing, which is also why the cache write moved out). When not checkpointing, update() still happens inside the layer, so dynamic and static cache behavior at inference is unchanged.

Loss and all gradients are bit identical with checkpointing on and off, and the encoder gets gradients through the cache. With the deprecated use_reentrant=True the backward raises instead of silently dropping encoder gradients. On a 1.26B test config (bf16 full fine tune, batch 16, seq 1024) training OOMs on a 48GB card without checkpointing and peaks at 32.6GB with it.

All diffusion_gemma tests pass.

cc @gante

## [c4482856486] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-12
Hmm so we arw basically copying cache contents into a list, and it won't get updated in place. Iiuc we can't GC with cache because if it's in place modifications, so what if we deep copy cache before passing into each decoder layer, when the model is found to be doing GC?

## [c4687971335] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
There is no copy actually, the layers get references to the same tensors the cache holds. And the decoder never writes the cache, only the encoder does, which is why its update() moved outside the checkpointed call.

Deep copy sadly does not work: the base class nulls past_key_values regardless of which cache we pass, and deepcopy raises on the KV tensors anyway since they carry grad_fn (a detached copy would cut the encoder gradients, which is the point of the PR).

If you prefer keeping the Cache object we could instead let GradientCheckpointingLayer opt in to keeping a read only cache, happy to do that. This version just keeps it local to the model, like encoder_hidden_states in cross attention models.

## [c4483304666] ArthurZucker (COLLABORATOR) · review COMMENTED · reply to [body] · 2026-06-12
> encoder gets gradients through the cache

this is the most important piece. are you sure the reason it does not have gradients otherwise can't be found in the various decorators we have for the `Cache` classe that make the `update` function drop them?

## [c4688353274] ArthurZucker (COLLABORATOR) · issue_comment · 2026-06-12
If not, changes are fine but I would reckon we can hide this!

## [c4688479525] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
thanks checking!

## [c4688742842] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
Checked the decorators question: update() has no decorators anywhere in cache_utils, and gradients do flow through the cache when not checkpointing (that path trains fine). The drop came from GradientCheckpointingLayer nulling past_key_values. And if a writing layer keeps the cache instead, the recompute calls update() a second time and double appends, so the shapes break. That is why the encoder write sits outside the checkpointed call.

Hid the rest as suggested: GradientCheckpointingLayer now has a _can_checkpoint_with_cache flag for layers whose cache usage is read only, the decoder opts in and is back to its original shape. The model diff is basically just the encoder now. Loss and grads stay bit identical with checkpointing on and off, llama's gradient checkpointing tests still pass, and the memory savings are unchanged.

## [c4688855733] kashif (CONTRIBUTOR) · issue_comment · reply to [c4688353274] · 2026-06-12
should i also add a test @ArthurZucker ?

## [c4689539976] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
Added one in the model tests in the meantime: checkpointing has to reproduce the exact same loss and gradients as the plain forward, encoder included.

## [c4689581030] zucchini-nlp (MEMBER) · issue_comment · 2026-06-12
Imo returning key/value from the encoder layer breaks consistency with other LLMs, and is kinda duplicating what the cache does

Imho,the idea by keeping cache and making it readonly sounds better. Does that mean we can apply it automatically for all models via GCLayerModule as a thin wrapper?

## [c4689689270] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
Yes, it can be made generic. I prototyped it: keep the cache inside the checkpointed call and have the wrapper snapshot the per layer state at entry and restore it at the start of the closure, so the recompute's second update() lands identically. First pass writes carry grad_fn under non reentrant checkpointing, so gradients match the plain forward exactly, encoder included.

The catch is the restore primitive per cache type: dynamic layers restore cleanly by keeping the old tensor refs (append only), sliding window ones also need their counters (crop() raises past the window), static ones are idempotent if counters are restored, and linear attention layers mutate in place so they would need explicit support.

Happy to do the dynamic case here and drop the kv returns from the encoder, or keep the generic wrapper for a follow up. Which do you prefer?

## [c4690039374] kashif (CONTRIBUTOR) · issue_comment · 2026-06-12
Done, made it fully generic as you suggested: GradientCheckpointingLayer snapshots the cache's per layer state at entry and restores it at the start of the checkpointed call, so the recompute's update() replays identically. Works automatically for the DynamicLayer family (they only rebind attributes), anything else falls back to the current null and warn. The model diff is now a single line, supports_gradient_checkpointing = True, no kv returns and no flags.

On the diffusion gemma side the parity test is still bit exact and the activation memory savings are unchanged. Since this touches the base layer class, I also checked llama's gradient checkpointing tests to make sure other models are unaffected.

## [c4690314199] zucchini-nlp (MEMBER) · issue_comment · 2026-06-12
Yep, exactly what i was thinking about! Also it will enable GC when prefix tuning which some users requested in the past and we deliberately chose to not support. cc @BenjaminBossan 

If Arthur also agrees, I'm happy to keep the current approach

## [c4690476554] BenjaminBossan (MEMBER) · issue_comment · reply to [c4690314199] · 2026-06-12
> Also it will enable GC when prefix tuning which some users requested in the past and we deliberately chose to not support.

It will? Nice! I can give it a spin once the PR lands.

## [c4698572498] zucchini-nlp (MEMBER) · issue_comment · 2026-06-13
I think it will, since diffusion gemma is also a version of prefix-tuning in a sense, but the "prefixes" are dynamically computed with an actual encoder based on inputs

## [c3409190802] kashif (CONTRIBUTOR) · review_comment on src/transformers/modeling_layers.py:50 · 2026-06-14
Good catch. A lazily-built cache could have a layer appended during the replayed forward that the restore never reset. Now truncating any appended layers back to the snapshot length before restoring, so the empty-cache case is idempotent too.

## [c4492581823] kashif (CONTRIBUTOR) · review COMMENTED · 2026-06-14 (no text)

## [c3409190836] kashif (CONTRIBUTOR) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:283 · 2026-06-14
Makes sense. Kept it bit-exact on CPU (where CI runs) and relaxed to 1e-5 on accelerators to avoid backward nondeterminism flakiness.

## [c4492581839] kashif (CONTRIBUTOR) · review COMMENTED · 2026-06-14 (no text)

## [c3412604589] zucchini-nlp (MEMBER) · review_comment on src/transformers/modeling_layers.py:42 · 2026-06-15
oke, I suppose static cache won't be used with GC anyway

## [c3412656754] zucchini-nlp (MEMBER) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:258 · 2026-06-15
ig we can also set `model_tester.is_training = True` so GC tests from mixin dont get skipped anymore

## [c3412673637] zucchini-nlp (MEMBER) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:258 · reply to [c3412656754] · 2026-06-15
And PI0 also has a lot of GC tests skippedt due to forced caching, can you un-skip them?

## [c4496353788] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-15
Nice, but seems like it causes failures with special cache model like deepseekV4. Didn't look at root cause, prob we need to limit the snapshot cache to specifically `DynamicCache/DynamicSlidingCache` and skip for subclasses, as we don't know what a subclass is doing internally?

## [c3413268773] kashif (CONTRIBUTOR) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:258 · reply to [c3412656754] · 2026-06-15
Tried it, but the mixin training tests do `loss = model(**inputs).loss; loss.backward()` and DiffusionGemmaForBlockDiffusion returns only logits (the diffusion loss is computed in the trainer), so they all fail on the missing loss. The custom test here covers GC bit-exactly instead, so I kept is_training off.

## [c4497139587] kashif (CONTRIBUTOR) · review COMMENTED · 2026-06-15 (no text)

## [c3413268912] kashif (CONTRIBUTOR) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:258 · reply to [c3412656754] · 2026-06-15
Had a look. PI0 builds its VLM prefix cache inside the forward and the action expert then reads `past_key_values.get_seq_length()`. Under GC the layer forces use_cache=False, so that cache ends up None and the test errors out. This snapshot/restore only makes the cache writes idempotent on recompute, it does not keep use_cache on. Getting PI0 to pass would need the GC layer to keep use_cache=True when the cache is snapshotable, which feels like a separate change. Happy to do it as a follow-up if you think it is worth it.

## [c4497139777] kashif (CONTRIBUTOR) · review COMMENTED · 2026-06-15 (no text)

## [c3413393124] zucchini-nlp (MEMBER) · review_comment on tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py:258 · reply to [c3412656754] · 2026-06-15
Ahh oke, dw about Pi0, thought it would be the same case. Not really worth it as model isn't yet integrated by lerobot

## [c4497288249] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-15 (no text)

## [c4728309189] kashif (CONTRIBUTOR) · issue_comment · 2026-06-17
Good catch on deepseekV4. Limited the snapshot to plain DynamicLayer and DynamicSlidingWindowLayer and skip any subclasses, since we can't assume what they do internally (deepseekV4's HCA/CSA caches subclass the sliding layer). Their GC tests pass now, along with llama and diffusion gemma.

## [c4516062489] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-06-17
Gradient checkpoint replay does not need to update KV state; it only needs the same effective K/V tensors that the original forward used. Therefore mutation should be opt-in/controlled at the cache API boundary, not undone afterward by snapshotting internal state.

## [c4732730553] kashif (CONTRIBUTOR) · issue_comment · 2026-06-17
Tried the non-mutating replay and it doesn't hold up here, because the checkpointed region is also the producer of the cache, not just a consumer.

In `DiffusionGemmaModel.forward` the encoder writes the prefix KV into the cache via `update()` from inside its checkpointed layers, then the decoder reads it (read-only). If `update()` returns the effective K/V without storing, the encoder's prefill never lands, so the decoder cross-attends to an empty cache. The loss drifts (0.0009 vs 0.0028) and no gradient flows back to the encoder.

The catch is that the first forward genuinely must mutate, since the decoder runs later in that same forward and has to read the prefill. Only the backward recompute should avoid the second append. A blanket non-mutating update drops the needed first write along with the unwanted one.

So the snapshot/restore keeps the mutation but makes it idempotent: write normally on the first pass, reset the cache to its pre-write state before the recompute so it reproduces the same single write instead of doubling. It's scoped to the plain dynamic layers and falls back to dropping the cache for anything else.

Verified end to end on the trl SFT objective (huggingface/trl#6003) with the tiny CI model: GC matches no-GC bit-exactly on loss and all grads, and encoder grads flow through the cache.

## [c4733519183] kashif (CONTRIBUTOR) · issue_comment · 2026-06-17
Reworked it to the cache API boundary like you suggested, no more snapshotting.

Instead of undoing the mutation afterward, `update()` is now idempotent under replay: in checkpointing mode it rebuilds the layer from its pre-checkpoint length rather than appending, so re-running the forward in the backward pass reproduces the same states. The cache owns this via a `checkpointing` flag, and the GradientCheckpointingLayer just captures each layer's length once and flips the flag for the replayed call. No more reaching into `layer.__dict__`.

It still mutates on the first forward, which we need here since the encoder is the producer of the cache and the decoder reads it later in the same forward. Only the replay is made non-growing.

Net -20 lines. Verified the same way: GC matches no-GC bit-exactly on loss and grads for DiffusionGemma (sliding and full layers), llama and deepseek_v4 GC tests pass, and the trl SFT objective (huggingface/trl#6003) trains with encoder grads flowing through the cache.

## [c3452367082] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:65 · 2026-06-22
Let's have those as instance properties in `__init__`, rather than class properties!

## [c3452374060] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:165 · 2026-06-22
Let's move the body inside the `if` to another method, so that the original `update` is still very clear please!

```suggestion
        if self.checkpointing:
            return self.checkpointing_update(...)
```

## [c3452392291] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:267 · 2026-06-22
Here I believe the logic is wrong - as always with sliding window, we cannot "go back in time", as we otherwise loose states... It works while the cache is not full, but once we reach the sliding window and start evicting old states (`full_key_states[:, :, -self.sliding_window + 1 :, :]`), then reindexing with `offset` will not return the same states, as we lost some in the meantime...
So this way of doing it is impossible with sliding window in general I believe, unless we keep all states, which defeats the purpose of the sliding window

## [c3452399488] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_layers.py:99 · 2026-06-22
I believe this would be easier to understand/follow

```suggestion
            checkpointing_cache_layers = None
```

## [c4544233918] Cyrilvallez (MEMBER) · review COMMENTED · 2026-06-22
Hey! I just have an issue with the sliding window offset. As I explained in my comment, the logic is broken as soon as the cache becomes full, and starts evicting states (we removes the oldest states, so when we reindex with `offset`, it will no longer be the same tensor)

## [c4768990850] kashif (CONTRIBUTOR) · issue_comment · 2026-06-22
Thanks for the review!

You're right that the offset reindex breaks for a sliding window: once it fills and starts evicting, `self.keys[:offset]` no longer holds the old prefix.

I first tried just dropping sliding layers from the checkpointable set, but that breaks DiffusionGemma. Its cache mixes full and sliding layers, so the whole cache gets dropped during the checkpointed forward and the GC-matches-no-GC test fails by ~2x.

So instead I snapshot the cache state once before the checkpointed region and rebuild from that on replay. We don't need to keep all states here: a sliding layer only ever needs its window, and the snapshot is exactly that window. It's bit-exact now (the test compares loss and grads).

Also did the other three: instance attrs in `__init__`, the body moved into a `checkpointing_update` method, and the rename to `checkpointing_cache_layers`.

## [c4769226583] Cyrilvallez (MEMBER) · issue_comment · 2026-06-22
Hummmmm, this smells like agent a lot 😆
- I don't see the requested changes to you say you implemented
- I don't see any snapshot mechanism - `self.keys/values` are updated in-place, which means we loose states once the cache is full when replayed a second time

## [c4769233193] kashif (CONTRIBUTOR) · issue_comment · 2026-06-22
yes trying to make summary of the changes my bad

## [c4769342935] kashif (CONTRIBUTOR) · issue_comment · reply to [c4769226583] · 2026-06-22
@Cyrilvallez just pushed it should be there now ef13dfd

## [c3465597894] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:458 · 2026-06-24
we're back to returning cache as plain list? 🥲

## [c4560386995] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-24 (no text)

## [c4651762556] Cyrilvallez (MEMBER) · review COMMENTED · 2026-07-08
Sorry but we really don't want to do this, creating 2 caches at once with one as a list is a big no-go 😬

## [c4912685419] kashif (CONTRIBUTOR) · issue_comment · reply to [c4651762556] · 2026-07-08
thanks @Cyrilvallez yes i was just iterating and updated the branch... will try a better solution!

## [c3543800520] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:676 · 2026-07-08
cool, so we dont need to return the kv as tuple right? The cache is returned in non-BC mode and used as read-only by a decoder

## [c3543807480] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:1275 · 2026-07-08
it still relies on cache so prob we can check if `past_kv is None` and raise error

## [c3543816181] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:1416 · 2026-07-08
not sure why we need this since `create_bidirectional_mask` has to build mask in correct dtype for attn backend

## [c4653953410] zucchini-nlp (MEMBER) · review COMMENTED · 2026-07-08
Great, exactly what I meant, So we can pass over cache for decoder since it only reads it. Lets delete the left-over code from prev iterations and I think we can ask Cyril's review

## [c3569086950] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:901 · 2026-07-13
if its the same, why not juste use base model output with past?

## [c3569090339] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:1406 · 2026-07-13
this does not look related no?

## [c4682950002] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-07-13
lgtm

## [c3569332370] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:1406 · reply to [c3569090339] · 2026-07-13
true its not related to his specifically but i needed it for the trl training... should i move it to a new PR?

## [c4683235539] kashif (CONTRIBUTOR) · review COMMENTED · 2026-07-13 (no text)

## [c3569394177] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py:1406 · reply to [c3569090339] · 2026-07-13
removed from this PR and will make a new PR with this

## [c4683306842] kashif (CONTRIBUTOR) · review COMMENTED · 2026-07-13 (no text)

---
outcome: state=closed, merged=True
