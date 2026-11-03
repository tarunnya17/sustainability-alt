# #45847 [pull_request] Streamable chat parsing
https://github.com/huggingface/transformers/pull/45847  ·  opened by Rocketknight1 on 2026-05-08

## [body] Rocketknight1 (MEMBER) · opening post · 2026-05-08
Work in progress PR for streamable chat parsing. This introduces a new improved kind of response schema, which I'm calling `response_template`. Response templates are a lot easier to write, since they avoid most of the messy regexes of the older system. More importantly, though, using a region-based system means that model authors can write templates that enable streaming parsing, emitting fields as they're ready, rather than needing the whole message to be ready at once.

TODO:

- [x] Merge/cleanup content types
- [x] Can we merge `coerce` into content types?
- [x] Correct handling of tokens/strings inputs, batched lists, etc.
- [x] Final docs cleanup, ensure examples match API
- [x] Lots and lots of general testing
- [x] Handling of multiple tool calls - do we need to emit arrays differently?
- [x] Explicit option for "multiple literals" as a match without needing a regex, avoids regex buffer
- [x] Properly document the weird content args

## [c4461472910] Rocketknight1 (MEMBER) · issue_comment · 2026-05-15
This should now be ready for review! It's a lot, but thankfully I think you can skim a lot of it. It's also a little ugly right now, because we need to maintain legacy support for the older `parse_response()` method, but once this is in we can begin deprecating that very quickly. That cleanup, and support for batched inputs, will come in a follow-up PR (batched handling will also be needlessly messy because of the legacy code right now, but it'll be much cleaner to add once it's removed)

The main things to review are:
- The doc
- The public-facing changes in `response_parser.py`, `processing_utils.py` and `tokenizer_utils.py`
- The test coverage

Most other files are agent-written boilerplate changes to support the new API (the pipeline + Gemma conversion script changes), or agent-written internal mechanics that are mostly dense and should be well-covered by tests (`content_parsers.py`, `response_templates.py`  and the second half of `response_parser.py`)

## [c3258455998] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:47 · 2026-05-18
```suggestion
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(model.device)
```

Also, reminder to use a model or revision with a valid `response_template`

## [c3258524574] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:85 · 2026-05-18
```suggestion
The parser will emit **events** as text from the generation process is fed in. This indicates which region is currently being generated. When
```

Trying to clarify the (usual) relationship with generation; perhaps we could provide a combined example that shows streaming generation with response parsing.

## [c3258583200] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:129 · 2026-05-18
In that case we should update the prompt and pass the tools so it triggers the greet_user call, just as it's done later in the advanced usage section.

## [c3258612649] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:236 · 2026-05-18
```suggestion
Finally, you can specify `optional: false` for fields that must be present. If parsing finishes and a non optional field
```

## [c3258621570] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:270 · 2026-05-18
😂

## [c3258663119] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:323 · 2026-05-18
Maybe mention the `transformers["chat_template"]` extra that includes this?

## [c3258700453] pcuenca (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:93 · 2026-05-18
```suggestion
            "transform": "{type: 'function', function: {name: name, arguments: content}}",
},
```

I think this became a jmespatch `transform` string

## [c4309667049] pcuenca (MEMBER) · review COMMENTED · 2026-05-18
Quick look at the docs + gemma 4 testing.

## [c3259569597] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:93 · reply to [c3258700453] · 2026-05-18
Ah, you're totally right! I missed that one after deprecating `assemble`

## [c4310919386] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-18 (no text)

## [c3259611941] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:323 · reply to [c3258663119] · 2026-05-18
Done!

## [c4310964974] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-18 (no text)

## [c3259672080] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:129 · reply to [c3258583200] · 2026-05-18
Fixed the text! I don't want to make the example too complex by adding tool definitions, so I just changed the text so I don't say the event stream matches the example above.

## [c4311035143] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-18 (no text)

## [c3260036325] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:129 · reply to [c3258583200] · 2026-05-18
Sounds good!

## [c4311443590] pcuenca (MEMBER) · review COMMENTED · 2026-05-18 (no text)

## [c3266233819] ariG23498 (CONTRIBUTOR) · review_comment on docs/source/en/chat_response_parsing.md:270 · reply to [c3258621570] · 2026-05-19
This comment should never be resolved. LOVE IT. 😂

## [c4318844088] ariG23498 (CONTRIBUTOR) · review COMMENTED · 2026-05-19 (no text)

## [c3266953592] pcuenca (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · 2026-05-19
Note this is Python-only syntax for named groups, so we'd need a translation layer to ICU somewhere for downstream implementations, cc @xenova. Perhaps we should accept both in transformers and document ICU as the standard?

## [c4319750545] pcuenca (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c3267068331] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · reply to [c3266953592] · 2026-05-19
Ah, I wasn't aware! Is there a universal way to do it, or is there no universal way to make named groups work? I might even prefer unnamed groups and putting the names in a separate field rather than causing cross-language problems

## [c4319888484] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c3267158308] pcuenca (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · reply to [c3266953592] · 2026-05-19
I didn't find a universal way to do it, unfortunately. [This is Python](https://docs.python.org/3/library/re.html#regular-expression-syntax) and this is ICU (screenshots with the relevant fragments below).

Yeah, perhaps unnamed groups is the best option if it doesn't make things too confusing.

[image]
[image]

## [c4319990909] pcuenca (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c3267394058] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · reply to [c3266953592] · 2026-05-19
Hmmn, it might be possible to handle this universally by just having other language implementations strip the `<>` angle brackets if their regex engine treats them as part of the name!

## [c4320274130] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c3267688891] pcuenca (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · reply to [c3266953592] · 2026-05-19
The main problem is the `P`, the regex has to be rewritten without.

## [c4320652846] pcuenca (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c4321179215] pcuenca (MEMBER) · review COMMENTED · 2026-05-19 (no text)

## [c3268081431] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:341 · 2026-05-19
It's more about the `P` in `?P<` vs `?<`, [this is what I'm currently doing](https://github.com/huggingface/swift-transformers/blob/48b3171cf28fbf025ee77709d4f18319ec994316/Sources/Tokenizers/ResponseTemplate.swift#L268-L273).

## [c3274190057] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:87 · reply to [c3266953592] · 2026-05-20
Ah, I'm sorry, I misread that bit! I'll update the docs

## [c4328784434] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-20 (no text)

## [c3274867691] pcuenca (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:463 · 2026-05-20
Perhaps we could show a fragment of a rendered Cohere tool call so readers understand it contains `tool_name` and `parameters`.

## [c4329561360] pcuenca (MEMBER) · review COMMENTED · 2026-05-20 (no text)

## [c3281477921] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:341 · reply to [c3268081431] · 2026-05-21
Should be resolved in the docs now!

## [c4337269898] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-21 (no text)

## [c3304465915] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:463 · reply to [c3274867691] · 2026-05-26
Added a little example to clarify!

## [c4364558993] Rocketknight1 (MEMBER) · review COMMENTED · 2026-05-26 (no text)

## [c4660980383] Rocketknight1 (MEMBER) · issue_comment · 2026-06-09
I think we've gotten all the reviews we're going to get! Pinging core maintainers @arthurzucker @cyrilvallez @vasqu if any of you wants a veto before I merge this :sweat_smile:

## [c3436624350] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:212 · 2026-06-18
example / walkthrough would help!

## [c3436634196] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:271 · 2026-06-18
small examples here again can help otherwise its hard to visualize

## [c3436644938] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:301 · 2026-06-18
i would put small example for each type here

## [c3436656183] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:407 · 2026-06-18
i would show again how that would render

## [c3436664686] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:438 · 2026-06-18
again would like to see example!!!

## [c3436667699] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:438 · reply to [c3436664686] · 2026-06-18
like out -> becomes what with this

## [c3436672496] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:474 · 2026-06-18
is the output the string? model outputs JSON like right?

## [c3436677219] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:488 · 2026-06-18
very nice

## [c3436691060] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:7 · 2026-06-18
half of the licence is missing!

## [c3436694956] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:46 · 2026-06-18
if anything is vendored let's credit !

## [c3436700358] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:140 · 2026-06-18
can people easily register custom parsers?

## [c3436722198] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:28 · 2026-06-18
lits vs lists? IDK if its intended, idk what a lits is

## [c3436724089] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:28 · reply to [c3436722198] · 2026-06-18
literals? 
re is regex? 
can we get full names?

## [c3436738782] ArthurZucker (COLLABORATOR) · review_comment on tests/utils/test_chat_parsing.py:183 · 2026-06-18
these are very nice!

## [c4525874059] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-06-18
TY! 
IDK how much we can trust AI for the core part, I don't have the time to deep dive into what it's doing. The format is the most important, we can improve what's inside later on! 

Examples are really important when defining the standards imo! Let's make sure we add more when explaining. 

`"content": "text",` appears a lot, could be default in lots of cases no? 
But that's a nit. 
Format wise I like json, I think it should be alright to express stuff. Cross pateform is super important so it helps. !

## [c3438502186] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:206 · 2026-06-18
I think this message about multi-turn belongs in the initial context as well. This was kind of what I was wondering about reading from top to bottom

## [c3438514378] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:230 · 2026-06-18
Hmm might be a dumb question but why was the chat template jinja but this response parser json? Do we not introduce attack vectors with any simple json?

## [c3438549365] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:495 · 2026-06-18
This feels personal 😆

## [c3438567658] vasqu (COLLABORATOR) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:1 · 2026-06-18
So we are starting with gemma4 as the first one?

## [c3438575321] vasqu (COLLABORATOR) · review_comment on src/transformers/pipelines/image_text_to_text.py:464 · 2026-06-18
So this is BC only for gemma I guess? Maybe we should be a bit aggressive here and deprecate this as fast as we can? Before the schema becomes an annoying relict?

## [c3438577245] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:7 · reply to [c3436691060] · 2026-06-18
Happens above as well, would double check across the files

## [c3438596135] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:77 · 2026-06-18
_text(text, args) no?

## [c3438606291] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:104 · 2026-06-18
imo typing would be nice here at least. we rely on heavy workload on what we expect

## [c3438614368] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:155 · 2026-06-18
If we go into the direction of being extensible, we would need to change the logic here potentially?

## [c3438624922] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:214 · 2026-06-18
why shorten it? fld -> field

## [c3438629630] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:217 · 2026-06-18
could it be a dict or something iterable in general? ig we want to keep this simple?

## [c3438633104] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:19 · 2026-06-18
```suggestion
    and the prompt prefix and parses them without streaming any events, then returns the parsed message.
```

## [c3438635383] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:24 · 2026-06-18
```suggestion
    message, _ = stream.finalize()
```
nit

## [c3438637362] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:28 · reply to [c3436722198] · 2026-06-18
+1

## [c3438648137] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:77 · 2026-06-18
```suggestion
    literal strings otherwise; `can_extend` is `True` if one literal in the list is a strict prefix of
```

## [c3438652327] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:80 · 2026-06-18
same goes here for shortening imo. no need to shorten, rather be explicit so no one has to rethink what it means

## [c3438659680] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:97 · 2026-06-18
wait how does this work exactly? is this our internal pattern we keep up with? Just thinking because each tokenizer can have its own eos token(s)

## [c3438664049] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:142 · 2026-06-18
Feel like this could be a nice frozen set as well; might be interesting for developers to access for whatever reason

## [c3438687702] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:230 · 2026-06-18
This is pretty hard to read because it validates at each step, any way we could refactor this a bit? Not sure how exactly yet but there could be a more grouped flow? At the start we can read every attribute and have separate validation functions?

## [c3438694149] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:61 · 2026-06-18
nit feel like we can keep that public 🤔

## [c3438710059] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:91 · 2026-06-18
Doesn't really need to be its own function imo, we can just do it directly after the prefix. Or do you think the docs wont fit well otherwise?

## [c3438722290] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:97 · 2026-06-18
Makes me think, can we reset the complete state of the parser? For example if I simply want the same parser object but across different input streams? I.e. something async friendly. Likely harder to do so just a future thought but I think we can keep it in mind for the initial design as well

## [c3438737829] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:207 · 2026-06-18
no asserts please

## [c3438764662] vasqu (COLLABORATOR) · review_comment on tests/pipelines/test_pipelines_text_generation.py:297 · 2026-06-18
Just fmi, are overriding the templates to simplify or why is it necessary?

## [c3438767661] vasqu (COLLABORATOR) · review_comment on setup.py:192 · 2026-06-18
is it really related to this pr?

## [c3438776089] vasqu (COLLABORATOR) · review_comment on tests/utils/test_chat_parsing_utils.py:285 · 2026-06-18
That is a huge limitation imo. We really should aim to enable batching 🤔 maybe not with streaming at the same time (async) but at least in a simpler variation where we can safely process the complete text

## [c3438790320] vasqu (COLLABORATOR) · review_comment on tests/utils/test_chat_parsing.py:1 · 2026-06-18
They are currently not integrated into CI ig? Should we add them to slow ci? Just unsure how long these would take 👀 

In general no complaints here, solid integration tests

## [c4528082259] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-18
Sorry for the delays, also some comments from my side. My only real concern is the batching capability being a limitation now - I don't expect the stream based parsing to be fully async but it would be nice if we had some simple version for batches 🤔

## [c3443093583] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:230 · reply to [c3438514378] · 2026-06-19
No, Python's JSON parser is safe against untrusted input! I think most languages should have a JSON parser that's safe, or at least that can be configured to be safe. It should be a lot easier to secure than Jinja.

## [c4533568281] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443098896] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:474 · reply to [c3436672496] · 2026-06-19
Yeah, the model outputs a JSON-like string, and we parse it and convert it to dicts in the correct format to append to the chat.

## [c4533574435] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443101177] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:495 · reply to [c3438549365] · 2026-06-19
I still feel guilty because I didn't think too much about non-Python frameworks when I chose Jinja 😅

## [c4533576933] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443104983] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/gemma4/convert_gemma4_weights.py:1 · reply to [c3438567658] · 2026-06-19
Yeah, Gemma4 is the only model that has the old-style `response_schema`. I'm hoping for a very, very short deprecation period where we merge this PR, then move Gemma4 to a new `response_template`, then totally deprecate/remove the old code.

## [c4533581156] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443107819] Rocketknight1 (MEMBER) · review_comment on src/transformers/pipelines/image_text_to_text.py:464 · reply to [c3438575321] · 2026-06-19
Yeah, all of the `response_schema` stuff is getting deprecated ASAP. I'm going to move Gemma4 to the new format as soon as this is merged, then do an immediate deprecation.

## [c4533584493] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443116664] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:140 · reply to [c3436700358] · 2026-06-19
There isn't a facility for that right now! Custom parsers introduce remote code execution issues, and the range of parsers we have right now covers every case I know of. I think if a new model needs a new parser, we should just accept a PR to add a new parser to `transformers`, or to expand the args for an existing parser to support the new model. But hopefully that should be quite rare.

## [c4533596992] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443118910] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:155 · reply to [c3438614368] · 2026-06-19
No plans for extensibility at the moment, see [my comment above](https://github.com/huggingface/transformers/pull/45847/changes#r3443116664)!

## [c4533600185] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443135000] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:77 · reply to [c3438648137] · 2026-06-19
This one isn't a typo! `iff` = "if and only if"

## [c4533621738] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443299424] Rocketknight1 (MEMBER) · review_comment on tests/utils/test_chat_parsing_utils.py:285 · reply to [c3438776089] · 2026-06-19
This is very temporary! I was basically trying to avoid code clutter while we deprecate the old `response_schema`. After it's gone, clean batching is much simpler. However, if you prefer I can try to implement batching for both, do the deprecation, then simplify after we deprecate

## [c4533810906] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443303156] Rocketknight1 (MEMBER) · review_comment on setup.py:192 · reply to [c3438767661] · 2026-06-19
It is! The reason is that `jmespath` was only added for `response_schema`. I refactored a lot for the `response_template` code and we don't need it anymore, so it can go now.

## [c4533816221] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443314803] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:77 · reply to [c3438648137] · 2026-06-19
Aya, can I play my card "it's not my first language" :laughing:

## [c4533830022] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-19 (no text)

## [c3443316752] vasqu (COLLABORATOR) · review_comment on setup.py:192 · reply to [c3438767661] · 2026-06-19
Gotcha that's a nice side effect then

## [c4533832079] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-19 (no text)

## [c3443321441] vasqu (COLLABORATOR) · review_comment on tests/utils/test_chat_parsing_utils.py:285 · reply to [c3438776089] · 2026-06-19
Yea I think, this is the biggest point to me. The rest are rather small issues or nits. But it really would be nice to support batching from the get-go even if it comes with some additional overhead at the beginning

## [c4533837244] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-19 (no text)

## [c3443482599] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:7 · reply to [c3436691060] · 2026-06-19
Yes, sorry! Fixed the licence in a few places, not sure how it got cut off like that

## [c4534021713] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443484323] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:46 · reply to [c3436694956] · 2026-06-19
None of this was copied from elsewhere, I think! A lot of it was agent-written because it was deep internals

## [c4534023579] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443485708] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:28 · reply to [c3436722198] · 2026-06-19
Did a lot of renaming / variable expanding!

## [c4534024953] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443529876] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:214 · reply to [c3438624922] · 2026-06-19
Fixed!

## [c4534075997] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443532064] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:77 · reply to [c3438596135] · 2026-06-19
Yes, good point. Fixed! This means the `strip()` arg is applied correctly

## [c4534079222] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443535335] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:24 · reply to [c3438635383] · 2026-06-19
Done!

## [c4534083715] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443536985] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:207 · reply to [c3438737829] · 2026-06-19
Fixed!

## [c4534086117] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443591147] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:230 · reply to [c3438687702] · 2026-06-19
Yeah, good idea! I refactored out the block that constructs a single field into its own method, which hopefully makes the overall flow way easier to follow.

## [c4534158099] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443622151] Rocketknight1 (MEMBER) · review_comment on tests/utils/test_chat_parsing.py:1 · reply to [c3438790320] · 2026-06-19
I think the CI is running these tests already! I think anything in `tests/utils/*` ends up in `tests_non_model`

## [c4534192913] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443637329] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:142 · reply to [c3438664049] · 2026-06-19
I'm hoping that this is going to be quite static! We can maybe move it somewhere accessible but I think it's fine as a constant for a single method

## [c4534209944] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443640391] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:97 · reply to [c3438722290] · 2026-06-19
Yeah, we can probably add this in future without breaking bc, so it's fine to leave for now! In general, though, I think it's fine to just make a new parser object when you need one

## [c4534213347] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443644135] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:61 · reply to [c3438694149] · 2026-06-19
I'm not sure! The `_spec` is basically the compiled spec, with compiled regexes and fields and so on. A lot of that is implementation details we might change later, and the only bit I want to commit to keeping constant for backward compatibility is the template itself + the core API like `parse_response()`

## [c4534217269] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443648407] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:97 · reply to [c3438659680] · 2026-06-19
"eos" here just means the end of whatever string we're parsing, not an EOS token! It's a bit confusing, though, but I wasn't sure how else to do it.

## [c4534221877] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443649250] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:80 · reply to [c3438652327] · 2026-06-19
Should be fixed!

## [c4534222774] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443651773] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:217 · reply to [c3438629630] · 2026-06-19
`transform_each` assumes we have a list we want to transform each element of, so I'm not sure how it would work with a `dict`. It's mostly used for `tool_calls` where the model might emit a list of multiple tool calls, and we want to do the same thing to each one

## [c4534225549] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443809953] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:212 · reply to [c3436624350] · 2026-06-19
Added!

## [c4534406107] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443810853] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:271 · reply to [c3436634196] · 2026-06-19
Scattered a few small examples here!

## [c4534406868] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443811221] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:301 · reply to [c3436644938] · 2026-06-19
Done!

## [c4534407244] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443846934] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:206 · reply to [c3438502186] · 2026-06-19
Amended the intro to explain this a bit better!

## [c4534446819] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443872217] Rocketknight1 (MEMBER) · review_comment on tests/pipelines/test_pipelines_text_generation.py:297 · reply to [c3438764662] · 2026-06-19
In this case I'm overriding the template to prefill part of the assistant message, to check that parsing still works!

## [c4534476403] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-19 (no text)

## [c3443882023] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:97 · reply to [c3438659680] · 2026-06-19
Ic, we should be careful that there is no potential clash with existing literals then?

## [c4534486522] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-19 (no text)

## [c3454259637] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:312 · 2026-06-22
😂 but we should refrain from formulating it as this lol

## [c3454296532] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:420 · 2026-06-22
wording :D abusing -> versatile

## [c3454307697] vasqu (COLLABORATOR) · review_comment on docs/source/en/chat_response_parsing.md:495 · 2026-06-22
nit: do we want to wrap this in a details / summary section

## [c3454322926] vasqu (COLLABORATOR) · review_comment on src/transformers/pipelines/any_to_any.py:448 · 2026-06-22
not really related to this PR but we should probably refactor these common parts out at some point 🤔 because we copy paste this through anything generation related

## [c3454332322] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:16 · 2026-06-22
```suggestion
and parses it into a single key in the output message dictionary. Functions are generally boilerplate."""
```
nit wouldnt really add this tbh, it helps as reviewer to take a double look but for a dev that just uses the code it's less useful imo

## [c3454343992] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:39 · 2026-06-22
Why not do _text(text, args) everywhere. It might seem a bit overkill but imo we should leave all the options open imo

## [c3454347789] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:56 · 2026-06-22
```suggestion
    string_delims = args.get("string_delims", [])
```
or any strong reason for the or?

## [c3454358708] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:73 · 2026-06-22
nit: We can partially precompile the regex?

## [c3454383395] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:141 · 2026-06-22
```suggestion
        line = _text(line, args)
        if not line or kv_sep not in line:
            continue
        k, v = line.split(kv_sep, 1)
        k, v = _text(k, args), _text(v, args)
        out[k] = _sub_parse(v, value_parser)
    return out
```
idk, I feel like we should reuse it where we can

## [c3454393603] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:175 · 2026-06-22
A bit hard to reread would try to itemize this into bullet points instead

## [c3454396680] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:172 · 2026-06-22
then you can reref the number here as small comment

## [c3454406728] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:197 · 2026-06-22
Imo can be simplified maybe? you essentially want to go through iterators for each value

## [c3454417391] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:33 · 2026-06-22
```suggestion
    open_regex: Any
```
similar below

## [c3454424176] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:69 · 2026-06-22
why not use somehing like logger.info instead? Dont see the need for a flag explicitly for this tbh

## [c3454424672] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:59 · 2026-06-22
nice doc

## [c3454429851] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:80 · 2026-06-22
nit, again more of a fan to use bullet points as it's easier to parse imo

## [c3454438002] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:103 · 2026-06-22
not on you but if python had some strong typing option to guarantee this 😢

## [c3454450364] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:103 · 2026-06-22
reopening re https://github.com/huggingface/transformers/pull/45847#discussion_r3443882023

Just a tad unsure about magic strings tbh

## [c3454461167] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:129 · 2026-06-22
this is weird, why do we even introduce a version? Shouldnt we be BC in most cases so the version should not matter?

## [c3454484577] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:219 · 2026-06-22
just fmi: should we limit the compile anchor to only use certain enums instead of raw strings? just so we have a quicker overview

## [c3454500815] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:23 · 2026-06-22
ok so here we are single stream dependent but on the tokenizer we do a for each (ig, not checked yet)? should we clarify this a bit in the docs

## [c3454509136] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:61 · reply to [c3438694149] · 2026-06-22
gotcha, then we can leave it private for now. no strong opinion

## [c3454514642] vasqu (COLLABORATOR) · review_comment on src/transformers/utils/chat_parsing/response_parser.py:91 · reply to [c3438710059] · 2026-06-22
rebump, altho if we add some state resetting etc this might be useful in itself

## [c3454554908] vasqu (COLLABORATOR) · review_comment on src/transformers/processing_utils.py:2234 · 2026-06-22
Hmm, shouldnt it still be the same type? It still nudges towards batched input no?

## [c3454561912] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:2070 · 2026-06-22
should we raise an error if we have both? or add prio and add a warning?

## [c3454569858] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:3343 · 2026-06-22
same but I feel like list[list[int]] should also be possible no? especially when we revert/default to legacy behavior if we dont have the new style

## [c3454575557] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:3391 · 2026-06-22
hmm, seems a bit brittle as check no?

## [c3454583037] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:3419 · 2026-06-22
its the same as for responses nit maybe to refactor a bit? but tbh very big nit

## [c3454589975] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:3447 · 2026-06-22
template

## [c3454600214] vasqu (COLLABORATOR) · review_comment on src/transformers/tokenization_utils_base.py:3432 · 2026-06-22
ok yea it's a foreach as we heavily rely on the stream parser. idk how chat templates are handled currently but let's aim to make this efficient in follow up PRs, e.g. some multi threading pool, async streamer etc

## [c3454611580] vasqu (COLLABORATOR) · review_comment on tests/utils/test_chat_parsing.py:171 · 2026-06-22
you prolly already thought of this but let's make these be loaded with auto tokenizer when we upload the templates one by one

## [c4546631235] vasqu (COLLABORATOR) · review APPROVED · 2026-06-22
Ok, I think this looks pretty ready. We can push for efficiency in follow up PRs imo (meaning for batching) but I think this is probably still gonna see a lot of changes regardless. Maybe we should note in the docs that this is experimental / in development so things are subjected to change?

My points are smaller in nature 🤗 nice job!

## [c3461922012] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:312 · reply to [c3454259637] · 2026-06-23
Fixed :cry:

## [c4555917383] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461922310] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:420 · reply to [c3454296532] · 2026-06-23
Fixed!

## [c4555917740] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461923824] Rocketknight1 (MEMBER) · review_comment on docs/source/en/chat_response_parsing.md:495 · reply to [c3454307697] · 2026-06-23
I think it's okay down at the bottom under the 'for developers' block, but we can change later if it's annoying!

## [c4555919474] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461928107] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:39 · reply to [c3454343992] · 2026-06-23
The only arg for _text is strip(), and int()/float() do that already, so it wouldn't really change much I think!

## [c4555924290] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461930979] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:73 · reply to [c3454358708] · 2026-06-23
Regex compilation is cached by default! The re/regex modules do it internally

## [c4555927518] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461939602] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:56 · reply to [c3454347789] · 2026-06-23
Fixed!

## [c4555938150] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461940150] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/content_parsers.py:141 · reply to [c3454383395] · 2026-06-23
Refactored!

## [c4555938827] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461945023] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:103 · reply to [c3454450364] · 2026-06-23
Dropped 'eos', and yeah, agreed. The system actually works fine if we close the last region on EOS regardless!

## [c4555944796] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461947136] Rocketknight1 (MEMBER) · review_comment on src/transformers/utils/chat_parsing/response_templates.py:129 · reply to [c3454461167] · 2026-06-23
I'm introducing it in case I need to break things in future! It's a just-in-case safety thing

## [c4555947232] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461952083] Rocketknight1 (MEMBER) · review_comment on src/transformers/tokenization_utils_base.py:2070 · reply to [c3454561912] · 2026-06-23
During the (hopefully short) changeover period we probably want to keep both, but prioritize the new format. As soon as we deprecate the old format we'll just stop saving it entirely even when we load it

## [c4555953125] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3461952969] Rocketknight1 (MEMBER) · review_comment on src/transformers/tokenization_utils_base.py:3391 · reply to [c3454575557] · 2026-06-23
Fixed!

## [c4555954105] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-23 (no text)

## [c3467561887] Rocketknight1 (MEMBER) · review_comment on src/transformers/tokenization_utils_base.py:3447 · reply to [c3454589975] · 2026-06-24
Done!

## [c4562754086] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-24 (no text)

## [c3467568983] Rocketknight1 (MEMBER) · review_comment on src/transformers/tokenization_utils_base.py:3432 · reply to [c3454600214] · 2026-06-24
I think chat templates are also just a `foreach` because there is no parallel Jinja! I'm hopeful that the region matching is pretty fast though, especially when run on a whole message at once.

## [c4562762476] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-24 (no text)

## [c3467570782] Rocketknight1 (MEMBER) · review_comment on src/transformers/tokenization_utils_base.py:3432 · reply to [c3454600214] · 2026-06-24
But yeah if performance is annoying later we can definitely refactor to speed it up a lot without breaking BC

## [c4562764520] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-24 (no text)

## [c4790004509] Rocketknight1 (MEMBER) · issue_comment · 2026-06-24
I think most of the review comments have now been addressed, going to merge! There are a few nits in docs or performance that we can move to follow-up work but nothing that will break BC, so I think merging is safe at this point!

---
outcome: state=closed, merged=True
