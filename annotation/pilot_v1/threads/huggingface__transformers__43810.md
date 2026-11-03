# #43810 [pull_request] add bidirectional attention to qwen and llama configs
https://github.com/huggingface/transformers/pull/43810  ·  opened by michaelfeil on 2026-02-06

## [body] michaelfeil (CONTRIBUTOR) · opening post · 2026-02-06
# What does this PR do?





Background: I want to remove custom code from some files, e.g. this one https://huggingface.co/nvidia/llama-embed-nemotron-8b/blob/main/llama_bidirectional_model.py or voyageai/voyage-4-nano model. 

They have custom code, which is not necessary.

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

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c3861994708] vasqu (COLLABORATOR) · issue_comment · 2026-02-06
Can you check #43705? It should enable your usecase just by passing `is_causal=False` as kwarg

## [c3862004328] michaelfeil (CONTRIBUTOR) · issue_comment · reply to [c3861994708] · 2026-02-06
@vasqu https://huggingface.co/nvidia/llama-embed-nemotron-8b/blob/main/llama_bidirectional_model.py + https://huggingface.co/nvidia/llama-embed-nemotron-8b/blob/main/config.json#L40 

Background: Embedding models are starting to use this flag more and more, but they are also forking the modeling code for the sake of having bidrectrional attention. Similar to gemma3, i want a custom flag.

## [c3862014964] vasqu (COLLABORATOR) · issue_comment · 2026-02-06
Does `is_causal` not suffice? You can either
- pass it as kwarg to the forward
- or set it in the config (and the forward afterwards will be adjusted accordingly)

I'd like to avoid causal specific flags, gemma3 is a bit of a special case. `is_causal` could be extended to most models

## [c3862026369] vasqu (COLLABORATOR) · issue_comment · 2026-02-06
I.e. it would only need to update the config to that of the respective llama and set `is_causal=False` as attribute and it will work out of the box without custom code

## [c3862030379] vasqu (COLLABORATOR) · issue_comment · 2026-02-06
But also cc @tomaarsen who wants to use this in ST

## [c3862328906] michaelfeil (CONTRIBUTOR) · issue_comment · 2026-02-06
also https://github.com/huggingface/text-embeddings-inference/pull/802 and https://github.com/huggingface/text-embeddings-inference/pull/805 for reference. 

Would be nice if, similar to gemma, `use_bidirectional_attention` would gain more traction, IMO, e.g. for TEI, this is hugley prefered. e.g. gemmaembeddings. Adding support for it rather sooner than later would be awesome!

## [c3862331233] michaelfeil (CONTRIBUTOR) · issue_comment · 2026-02-06
I'm mostly looking at it from a integration point into Rust, even if this is in python. I hope this makes @tomaarsen's job easier too + smoother integration into TEI. Embedding models are also super easy to serve with the wrong attention, e.g. the above models will load well, and TEI will just use the causal attention instead of full attention.

## [c3863283520] vasqu (COLLABORATOR) · issue_comment · 2026-02-07
I have the feeling you haven't really looked at #43705? 

Passing `is_causal` as kwarg or setting it as config attribute does everything for you:
- It exchanges the mask for a bidirectional one
- Overwrites the attention module's `is_causal` as kwarg takes precedence over the attr (there were a few prs to enable this) aka what you do by exchanging at init time (which is also brittle as we cannot dynamically change behavior).

Correctness has also been checked by checking such a remote code (with this new PR) to produce similar result (similar due to sdpa backends defaulting to either xformers or fa).

Hence, this is pretty much doing everything that you do here without modifying the modeling code. The only difference is that we use `is_causal` instead of `use_bidirectional_attention`. You can keep BC for remote code by just adding `is_causal: false` to the configs. 

The standard behind `is_causal` is to keep things as general as possible and make it extendable to future flags we can use for most models. `use_bidirectional_attention` is super specific to causal models and doesn't take into account 
1. That are more unique models like VLMs (paligemma's mask comes into my mind), encoder-decoder models etc
2. How specific and unscalable this is - every causal model that now subsequently gets a bidirectional model has to be modified after model addition.

The goal is to eventually refactor the gemma embedding model as well to follow the `is_causal` standard (with BC behavior).

## [c3863596353] michaelfeil (CONTRIBUTOR) · issue_comment · 2026-02-07
awesome, so we better add a PR to require latest transformers + `is_causal` to the config.json? I'm happy to contribute it, once we have a transformers version. Sounds great.

## [c3864021365] tomaarsen (MEMBER) · issue_comment · reply to [c3863596353] · 2026-02-07
> awesome, so we better add a PR to require latest transformers + `is_causal` to the config.json? I'm happy to contribute it, once we have a transformers version. Sounds great.

Bingo! In short, the PR @vasqu mentions was introduced exactly because we wanted it to be easier to turn a decoder into an encoder, as it's a very common strategy for modern embedding and retrieval models.

Most models should soon accept `is_causal=False` in either the config or the forward.

- Tom Aarsen

## [c3864034987] vasqu (COLLABORATOR) · issue_comment · 2026-02-07
Our minor release cycle should be much faster now so you hopefully won't have to wait long ;) also we should add more docs 😭

## [c3864922326] michaelfeil (CONTRIBUTOR) · issue_comment · 2026-02-07
This is awesome! Makes so much sense, thank you all!

---
outcome: state=closed, merged=False
