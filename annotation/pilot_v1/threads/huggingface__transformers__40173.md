# #40173 [pull_request] Chat Template Doc Fixes
https://github.com/huggingface/transformers/pull/40173  ·  opened by Rocketknight1 on 2025-08-14

## [body] Rocketknight1 (MEMBER) · opening post · 2025-08-14
The chat template docs were split into multiple parts and also refactored in recent PRs. The result is quite messy in some cases, and several examples are incomplete, or the text is misleading. I only noticed when I tried to refer to the docs while writing the `gpt-oss` template and realized that in many cases the text said things about chat templates that were completely incorrect or very unclear! This PR aims to fix it all up.

## [c3192012011] Rocketknight1 (MEMBER) · issue_comment · 2025-08-15
cc @stevhliu @zucchini-nlp I did a pass over all the chat template documentation including the multimodal templates! Mostly fixing errors or code that didn't work anymore, but also modernizing some parts and de-emphasizing things that aren't too relevant anymore. Let me know what you think!

## [c2281613157] zucchini-nlp (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:22 · 2025-08-18
maybe "should be a list containing multiple items"?

There are bits of code with assumption that the content is a list, only when `tokenize=True`

## [c3127246524] zucchini-nlp (MEMBER) · review APPROVED · 2025-08-18
Cool, happy to see less sub-sections in the docs. Just left one tiny comment in multimodal section :)

## [c2282596952] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:22 · reply to [c2281613157] · 2025-08-18
Yeah, good point. Made the change!

## [c3128652516] Rocketknight1 (MEMBER) · review COMMENTED · 2025-08-18 (no text)

## [c2288780765] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:19 · 2025-08-20
```suggestion
Chat models are conversational models you can send a message to and receive a response. Most language models from mid-2023 onwards are chat models and may be referred to as "instruct" or "instruction-tuned" models. Models that do not support chat are often referred to as "base" or "pretrained" models.
```

## [c2288793822] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:21 · 2025-08-20
```suggestion
Larger and newer models are generally more capable, but models specialized in certain domains (medical, legal text, non-English languages, etc.) can often outperform these larger models. Try leaderboards like [OpenLLM](https://hf.co/spaces/HuggingFaceH4/open_llm_leaderboard) and [LMSys Chatbot Arena](https://chat.lmsys.org/?leaderboard) to help you identify the best model for your use case.
```

## [c2288802387] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:57 · 2025-08-20
```suggestion
Chat models accept a list of messages (the chat history) as the input. Each message is a dictionary with `role` and `content` keys.
To start the chat, add a single `user` message. You can also optionally include a `system` message to give the model directions on how to behave.
```

## [c2288808289] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:80 · 2025-08-20
Maybe include a code example showing how to `.append()` the text to the `assistant` role so its super easy for users to just copy/paste it

```suggestion
To continue the conversation, you need to update the chat history with the model's response with one of the options below.

- Appending the text to the `assistant` role in `chat`.
- Reading `response[0]["generated_text"]`, which contains the full chat history, including the most recent response.
```

## [c2288816166] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:82 · 2025-08-20
```suggestion
Once you have the model's response, you can continue the conversation by appending a new `user` message to the chat history.
```

## [c2288835640] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:94 · 2025-08-20
```suggestion
By repeating this process, you can continue the conversation as long as you like until the context window or memory is exceeded.
```

## [c2288839354] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:98 · 2025-08-20
```suggestion
Transformers load models in full `float32` precision by default, and for a 8B model, this requires ~32GB of memory! Use the `torch_dtype="auto"` argument, which generally uses `bfloat16` for models that were trained with it, to reduce your memory usage.
```

## [c2288840689] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:103 · 2025-08-20
```suggestion
To lower memory usage even lower, you can quantize the model to 8-bit or 4-bit with [bitsandbytes](https://hf.co/docs/bitsandbytes/index). Create a [`BitsAndBytesConfig`] with your desired quantization settings and pass it to the pipelines `model_kwargs` parameter. The example below quantizes a model to 8-bits.
```

## [c2288845533] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:114 · 2025-08-20
```suggestion
This is a bottleneck for LLM text generation and the main options for improving generation speed are to either quantize a model or use hardware with higher memory bandwidth. Adding more compute power doesn't meaningfully help.
```

## [c2288848195] stevhliu (MEMBER) · review_comment on docs/source/en/conversations.md:119 · 2025-08-20
I don't think this needs to be a `Tip`.

```suggestion
Mixture-of-Expert (MoE) models such as [Mixtral](./model_doc/mixtral), [Qwen2MoE](./model_doc/qwen2_moe), and [GPT-OSS](./model_doc/gpt-oss) have lots of parameters, but only "activate" a small fraction of them to generate each token. As a result, MoE models generally have much lower memory bandwidth requirements and can be faster than a regular LLM of the same size. However, techniques like speculative decoding are ineffective with MoE models because more parameters become activated with each new speculated token.
```

## [c2288856957] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:20 · 2025-08-20
```suggestion
The [chat basics](./conversations) guide covers how to store chat histories and generate text from chat models using [`TextGenerationPipeline`]. 

This guide is intended for more advanced users, and covers the underlying classes and methods, as well as the key concepts for understanding what's actually going on when you chat with a model.
```

## [c2289125320] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:29 · 2025-08-20
```suggestion
Chat models are trained on data formatted as a sequence of messages. The important thing to understand though is that the chat is still just a sequence of tokens like all causal language models. The `role` and `content` dictionaries are converted to a token sequence with control tokens like `<|user|>`, `<|assistant|>` or `<|end_of_message|>`. Control tokens help a model understand the chat structure.

There are many chat formats, and different models may use different formats or control tokens, even if they were fine-tuned from the same base model. Use the [`~PreTrainedTokenizer.apply_chat_template`] method to access a model's **chat template**, which indicates a model's expected chat format.

The example below shows how two models fine-tuned from the same Mistral-7B base model can have different chat templates.
```

## [c2289127638] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:71 · 2025-08-20
```suggestion
Mistral-7B-Instruct uses `[INST]` and `[/INST]` tokens to indicate the start and end of user messages, while Zephyr-7B uses `<|user|>` and `<|assistant|>` tokens to indicate speaker roles. This is why chat templates are important - with the wrong control tokens, these models would have drastically worse performance.
```

## [c2289128294] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:73 · 2025-08-20
```suggestion
## Using `apply_chat_template`
```

## [c2289135084] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:75 · 2025-08-20
```suggestion
The input to `apply_chat_template` should be structured as a list of dictionaries with `role` and `content` keys. The `role` key specifies the speaker, and the `content` key contains the message. The common roles are:

 - `user` for messages from the user
 - `assistant` for messages from the model
 - `system` for directives on how the model should act (usually placed at the beginning of the chat)
```

## [c2289136984] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:77 · 2025-08-20
```suggestion
[`apply_chat_template`] takes this list and returns a formatted sequence. Set `tokenize=True` if you want to tokenize the sequence.
```

## [c2289137385] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:102 · 2025-08-20
```suggestion
Pass the tokenized chat to [`~GenerationMixin.generate`] to generate a response.
```

## [c2289145927] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:124 · 2025-08-20
```suggestion
Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When using `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you're tokenizing later to avoid duplicating these tokens.
This isn’t an issue if `apply_chat_template(tokenize=True)` and it's usually the safer option.
```

## [c2289158147] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:130 · 2025-08-20
```suggestion
The [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) argument adds tokens to the end of the chat.

> [!TIP]
> Remember, underneath the chat abstractions, chat models are causal language models that continue a sequence of tokens.

This control token indicates it's the models turn to respond. If you don't include this token, the model may behave unexpectedly such as continuing a user's message instead of replying to it.
```

## [c2289161775] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:132 · 2025-08-20
```suggestion
Compare the chat formats below when `add_generation_prompt` is set to `False` and `True`.
```

## [c2289166548] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:147 · 2025-08-20
May be nice to wrap the output difference for `add_generation_prompt` with the `<hfoptions>` blocks so users can more easily switch between the two outputs and compare them

## [c2289168793] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:159 · 2025-08-20
```suggestion
When `add_generation_prompt=True`, `<|im_start|>assistant` is added at the end to indicate the start of an `assistant` message. This lets the model know an `assistant` response is next.
```

## [c2289176831] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:182 · 2025-08-20
```suggestion
[`TextGenerationPipeline`] sets [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) to `True` by default to start a new message. However, if the final message in the chat has the `assistant` role, it assumes the message is a prefill and switches to `continue_final_message=True`. This is because most models don’t support multiple consecutive assistant messages. To override this behavior, explicitly pass the [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) argument to the pipeline.
```

## [c2289177233] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating.md:185 · 2025-08-20
```suggestion
## Model training
```

## [c2289270146] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:19 · 2025-08-20
```suggestion
Multimodal chat models accept inputs like images, audio or video, in addition to text. The `content` key in a multimodal chat history is a list containing multiple items of different types. This is unlike text-only chat models whose `content` key is a single string.
```

## [c2289270465] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:22 · 2025-08-20
```suggestion
```

## [c2289274288] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:25 · 2025-08-20
```suggestion
In the same way the [Tokenizer](./fast_tokenizer) class handles chat templates and tokenization for text-only models, 
the [Processor](./processors) class handles preprocessing, tokenization and chat templates for multimodal models. Their [`~ProcessorMixin.apply_chat_template`] methods are almost identical.
```

## [c2289276913] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:27 · 2025-08-20
```suggestion
This guide will show you how to chat with multimodal models with the high-level [`ImageTextToTextPipeline`] and at a lower level using the [`~ProcessorMixin.apply_chat_template`] and [`~GenerationMixin.generate`] methods.
```

## [c2289280414] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:32 · 2025-08-20
```suggestion
[`ImageTextToTextPipeline`] is a high-level image and text generation class with a “chat mode”. Chat mode is enabled when a conversational model is detected and the chat prompt is [properly formatted](./llm_tutorial#wrong-prompt-format).
```

## [c2289281916] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:34 · 2025-08-20
```suggestion
Add image and text blocks to the `content` key in the chat history.
```

## [c2289283642] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:52 · 2025-08-20
```suggestion
Create an [`ImageTextToTextPipeline`] and pass the chat to it. For large models, setting [device_map=“auto”](./models#big-model-inference) helps load the model quicker and automatically places it on the fastest device available. Setting the data type to [auto](./models#model-data-type) also helps save memory and improve speed.
```

## [c2289284169] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:63 · 2025-08-20
```suggestion
```

## [c2289284857] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:65 · 2025-08-20
```suggestion
```

## [c2289285737] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:71 · 2025-08-20
```suggestion
## Using `apply_chat_template`
```

## [c2289299373] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:74 · 2025-08-20
```suggestion
Like [text-only models](./chat_templating), use the [`~ProcessorMixin.apply_chat_template`] method to prepare the chat messages for multimodal models. 
This method handles the tokenization and formatting of the chat messages, including images and other media types. The resulting inputs are passed to the model for generation.
```

## [c2289299776] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:76 · 2025-08-20
```suggestion
```

## [c2289300594] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:100 · 2025-08-20
```suggestion
Pass `messages` to [`~ProcessorMixin.apply_chat_template`] to tokenize the input content. Unlike text models, the output of `apply_chat_template`
contains a `pixel_values` key with the preprocessed image data, in addition to the tokenized text.
```

## [c2289300937] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:107 · 2025-08-20
```suggestion
```

## [c2289301745] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:113 · 2025-08-20
```suggestion
Pass these inputs to [`~GenerationMixin.generate`].
```

## [c2289304245] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:123 · 2025-08-20
```suggestion
The decoded output contains the full conversation so far, including the user message and the placeholder tokens that contain the image information. You may need to trim the previous conversation from the output before displaying it to the user.
```

## [c2289305875] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:17 · 2025-08-20
```suggestion
# Tool use
```

## [c2289308578] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:22 · 2025-08-20
```suggestion
Chat models are commonly trained with support for "function-calling" or "tool-use". Tools are functions supplied by the user, which the model can choose to call as part of its response. For example, models could have access to a calculator tool to perform arithmetic without having to it internally.
```

## [c2289310038] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:30 · 2025-08-20
```suggestion
When a model supports tool-use, pass functions to the `tools` argument of [`~PreTrainedTokenizerBase.apply_chat_template`].
The tools are passed as either a [JSON schema](https://json-schema.org/learn) or Python functions. If you pass Python functions,
the arguments, argument types, and function docstring are parsed in order to generate the JSON schema automatically.
```

## [c2289311574] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:34 · 2025-08-20
```suggestion
docstrings. Refer to the examples below for how to format a tool-ready function.
```

## [c2289320314] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:63 · 2025-08-20
```suggestion
You can optionally add a `Returns:` block to the docstring and a return type to the function header, but most models won't use this information. The parser will also ignore the actual code inside the function!

What really matters is the function name, argument names, argument types, and docstring describing the function's purpose
and the purpose of its arguments. These create the "signature" the model will use to decide whether to call the tool.
```

## [c2289321489] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:65 · 2025-08-20
```suggestion
## Tool-calling Example
```

## [c2289322392] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:67 · 2025-08-20
```suggestion
Load a model and tokenizer that supports tool-use like [NousResearch/Hermes-2-Pro-Llama-3-8B](https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B), but you can also consider a larger model like [Command-R](./model_doc/cohere) and [Mixtral-8x22B](./model_doc/mixtral) if your hardware can support it.
```

## [c2289322821] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:78 · 2025-08-20
```suggestion
Create a chat history.
```

## [c2289325071] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:88 · 2025-08-20
```suggestion
Next, pass `messages` and a list of tools to [`~PreTrainedTokenizerBase.apply_chat_template`]. Tokenize the chat and generate a response.
```

## [c2289327890] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:104 · 2025-08-20
```suggestion
The chat model called the `get_current_temperature` tool with the correct parameters from the docstring. It inferred France as the location based on Paris, and that it should use Celsius for the units of temperature.

A model **cannot actually call the tool itself**. It requests a tool call, and it's your job to handle the call and append it and the result to the chat history.
```

## [c2289330093] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:106 · 2025-08-20
```suggestion
Hold the call in the `tool_calls` key of an `assistant` message. This is the recommended API, and should be supported by the chat template of most tool-using models.
```

## [c2289335687] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:109 · 2025-08-20
```suggestion
> Although `tool_calls` is similar to the OpenAI API, the OpenAI API uses a JSON string as its `tool_calls` format. This may cause errors or strange model behavior if used in Transformers, which expects a dict.
```

## [c2289336868] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:112 · 2025-08-20
```suggestion
```

## [c2289337810] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:119 · 2025-08-20
```suggestion
Append the tool response to the chat history with the `tool` role.
```

## [c2289338994] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:125 · 2025-08-20
```suggestion
Finally, allow the model to read the tool response and reply to the user.
```

## [c2289342849] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:140 · 2025-08-20
```suggestion
> Although the key in the assistant message is called `tool_calls`, in most cases, models only emit a single tool call at a time. Some older models emit multiple tool calls at the same time, but this is a
> significantly more complex process, as you need to handle multiple tool responses at once and disambiguate them, often using tool call IDs. Please refer to the model card to see exactly what format a model expects for tool calls.
```

## [c2289344177] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:143 · 2025-08-20
```suggestion
## JSON schemas
```

## [c2289367474] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:147 · 2025-08-20
```suggestion
Another way to define tools is by passing a [JSON schema](https://json-schema.org/learn/getting-started-step-by-step).

You can also manually call the low-level functions that convert Python functions to JSON schemas, and then check or edit the generated schemas. This is usually not necessary, but is useful for understanding the underlying mechanics. It's particularly important
for chat template authors who need to access the JSON schema to render the tool definitions.
```

## [c2289369914] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:149 · 2025-08-20
```suggestion
The  [`~PreTrainedTokenizerBase.apply_chat_template`] method uses the [get_json_schema](https://github.com/huggingface/transformers/blob/14561209291255e51c55260306c7d00c159381a5/src/transformers/utils/chat_template_utils.py#L205) function to convert Python functions to a JSON schema.
```

## [c2289372931] stevhliu (MEMBER) · review_comment on docs/source/en/chat_extras.md:185 · 2025-08-20
```suggestion
You can also pass JSON schema dicts to the `tools` argument of [`~PreTrainedTokenizerBase.apply_chat_template`] instead of Python functions.
```

## [c3137510960] stevhliu (MEMBER) · review COMMENTED · 2025-08-20
Thanks so much for updating and fixing, this is super! Most of my comments are just nits/suggestions for making the doc shorter where possible. Feel free to pick and choose the ones you vibe with the most :)

I'll review `chat_templating_writing` later!

## [c2291421356] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_extras.md:30 · reply to [c2289310038] · 2025-08-21
i got oxford comma'd lol

## [c3141221877] Rocketknight1 (MEMBER) · review COMMENTED · 2025-08-21 (no text)

## [c2291576772] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_templating_multimodal.md:65 · reply to [c2289284857] · 2025-08-21
I get to keep one line of whimsy and I nominate this one

## [c3141451073] Rocketknight1 (MEMBER) · review COMMENTED · 2025-08-21 (no text)

## [c3211436446] Rocketknight1 (MEMBER) · issue_comment · reply to [c3137510960] · 2025-08-21
@stevhliu woah, thanks for the comprehensive review! I accepted most of them - there's a few where I really want to highlight some point and I might keep the verbosity or warning blocks or whatever, just because I know users get trapped on that stuff

## [c2291828083] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:19 · 2025-08-21
```suggestion
A chat template is a [Jinja](https://jinja.palletsprojects.com/en/stable/templates/) template stored in the tokenizer's [chat_template](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.chat_template) attribute. Jinja is a templating language that allows you to write Python-like code and syntax.
```

## [c2291828486] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:21 · 2025-08-21
```suggestion
```

## [c2291914179] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:35 · 2025-08-21
```suggestion
If you stare at this for a while, you should realize that this is actually very like Python, albeit with some strange
`{%-` syntax. The template iterates over a list of messages, and for each message, it prints the role and content of 
the message, followed by an end-of-sequence token. If `add_generation_prompt=True`, it adds 
the starting header for an assistant message to the end of the conversation.
```

## [c2291920679] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:39 · 2025-08-21
```suggestion
Load the written template as a string and assign it to the tokenizer's `chat_template` attribute. Once set, the template is used whenever you call [`~PreTrainedTokenizerBase.apply_chat_template`]. It is also saved
with the tokenizer whenever [`~PreTrainedTokenizer.save_pretrained`] or [`~PreTrainedTokenizer.push_to_hub`] is called. The template is saved in the `chat_template.jinja` file in the tokenizer directory. You can
edit this file directly to change the template, which is often easier than manipulating a template string.
```

## [c2291923360] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:44 · 2025-08-21
```suggestion
The easiest way to start writing Jinja templates is to refer to existing templates. Use `print(tokenizer.chat_template)` on any chat model to see the template it's using. Try starting with simple models that don't call any tools or support RAG because tool-use models can have very complex templates. Finally, take a look at the [Jinja documentation](https://jinja.palletsprojects.com/en/stable/templates/#synopsis) for more details about formatting and syntax.
```

## [c2291925945] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:50 · 2025-08-21
```suggestion
For multimodal templates, the `chat_template` attribute is set on the **processor**, not the tokenizer. The `content` key of a message is often a list of content dicts,
rather than just a single string. You may wish to check the type of each content item in the list, and handle it accordingly.
```

## [c2291930151] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:56 · 2025-08-21
```suggestion
Generally, the template should not directly access image or video data. This is normally handled by the processor after template rendering has finished. Instead,
your template should emit a single special token like `<|image|>` or `<|video|>` when it encounters image or video content.  The processor will
expand the single special token out into a sequence of image or video tokens later. The exact tokens to emit depends on the model you're working with. We strongly recommend loading an existing multimodal processor to see how it handles data.
```

## [c2291930634] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:58 · 2025-08-21
```suggestion
The example template below handles mixed image and text content.
```

## [c2291933580] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:87 · 2025-08-21
```suggestion
This multimodal template is very similar to the more simple template above, but it checks for `content` lists,
and iterates over them to render `<|image|>` tokens where necessary. This allows images to be inserted "into the flow"
of user text.

Not all models work this way - some may move all images to the end of the user message,
for example. The chat template should always match the format the model was trained with.
```

## [c2291941653] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:97 · 2025-08-21
```suggestion
Jinja prints any whitespace before or after a block of text. This can be an issue for chat templates because adding extra whitespace that was not present during model training can harm performance. To remove the whitespace, add `-` to the Jinja line syntax. This allows you to write your template with Pythonic indentation and linebreaks, without accidentally printing an indentation in the rendered output.

The example template below doesn't use `-`, resulting in extra whitespace being printed in the output.
```

## [c2291953292] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:105 · 2025-08-21
```suggestion
We strongly recommend using `-` to ensure only the intended content is printed.
```

## [c2291953942] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:116 · 2025-08-21
```suggestion
```

## [c2292013747] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:126 · 2025-08-21
```suggestion
The only constants in a template are the `messages` variable and the `add_generation_prompt` boolean. However, you have
access to **any other keyword arguments that are passed** to the [`~PreTrainedTokenizerBase.apply_chat_template`] method.

This provides flexibility and enables support for use-cases we may not have thought of while designing the spec. The most common additional variable is `tools`, which contains a list of tools in JSON schema format. Although you can use any variable name you like, we highly recommend sticking to convention and using `tools` for this purpose. This makes templates more compatible with the standard API.

You also have access to any tokens contained in `tokenizer.special_tokens_map`, which often includes special tokens like `bos_token` and `eos_token`. Access these directly by name, like `{{- bos_token }}`.

There are two callable functions available to you. To call them, use `{{- function_name(argument) }}`.
```

## [c2292017259] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:168 · 2025-08-21
```suggestion
[Tools](./chat_extras) are passed as Python functions or a JSON schema. When functions are passed, a JSON schema is automatically generated and passed to the template. When a template accesses the `tools` variable, it is always a list of JSON schemas.
```

## [c2292018815] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:172 · 2025-08-21
```suggestion
Even though a template always receive tools as a JSON schema, you may need to radically change this format when rendering them to match the format a model was trained with. For example, [Command-R](./model_doc/cohere) was trained with tools defined with Python function headers. The template internally converts JSON schema types and renders the input tools as Python headers.

The example below shows how a tool is defined in JSON schema format.
```

## [c2292019572] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:198 · 2025-08-21
```suggestion
An example of handling tool definitions in a chat template is shown below. The specific tokens and layouts should be changed to match the ones the model was trained with.
```

## [c2292021358] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:209 · 2025-08-21
```suggestion
In addition to rendering the tool definitions, you also need to render **tool calls** and **tool responses** in the template.

Tool calls are generally passed in the `tool_calls` key of an `"assistant”` message. This is always a list even though most tool-calling models only support single tool calls, which means the list usually only contains a single element.
```

## [c2292022442] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:236 · 2025-08-21
```suggestion
A common pattern for handling tool calls is shown below. You can use this as a starting point, but make sure you template actually matches the format the model was trained with!
```

## [c2292023081] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:249 · 2025-08-21
```suggestion
Tool responses are message dicts with the `tool` role. They are much simpler than tool calls, and usually only contain the `role`, `name` and `content` keys.
```

## [c2292023922] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:259 · 2025-08-21
```suggestion
Some templates may not even need the `name` key, in which case, you can write your template to only read the `content` key.
```

## [c2292026002] stevhliu (MEMBER) · review_comment on docs/source/en/chat_templating_writing.md:269 · 2025-08-21
```suggestion
Once a template is ready, set it to the `chat_template` attribute in the tokenizer and test it with [`~PreTrainedTokenizerBase.apply_chat_template`]. If it works as expected, then upload it to the Hub with [`~PreTrainedTokenizer.push_to_hub`].
```

## [c3141836684] stevhliu (MEMBER) · review APPROVED · 2025-08-21
Thanks again and sorry for all the comments! 😅 

Do you think it'd be helpful to mention somewhere that you can also inspect the chat template on the Hub?

[image]

## [c3214627927] Rocketknight1 (MEMBER) · issue_comment · reply to [c3141836684] · 2025-08-22
@stevhliu Included almost all of those suggestions, with a couple of tweaks! I'm going to merge this soon - because it's so big and unwieldy as a PR, loading it in Github is getting slow and I'm getting merge conflicts with main, so I'd prefer to finish it up ASAP. I might do a second pass for remaining minor cleanup next week, for things like your templates-on-the-hub suggestion.

---
outcome: state=closed, merged=True
