# #41178 [issue] Memory leak when using openai/clip-vit-base-patch32
https://github.com/huggingface/transformers/issues/41178  ·  opened by Adefey on 2025-09-26

## [body] Adefey (NONE) · opening post · 2025-09-26
### System Info

Model runs on CPU+RAM. Ryzen 6800H, 14 GB DDR5, Host system: Fedora 41, Docker base image: python:3.13.7

FastAPI microservice is deployed in docker with these requirements:

fastapi==0.117.1
huggingface-hub==0.35.1
numpy==2.3.3
pillow==11.3.0
pydantic==2.11.9
python-multipart==0.0.20
requests==2.32.4
sentencepiece==0.2.1
torch==2.8.0
torchvision==0.23.0
transformers==4.56.2
uvicorn==0.37.0

transformers env:

embedding-service       | - `transformers` version: 4.56.2
embedding-service       | - Platform: Linux-6.16.7-100.fc41.x86_64-x86_64-with-glibc2.41
embedding-service       | - Python version: 3.13.7
embedding-service       | - Huggingface_hub version: 0.35.1
embedding-service       | - Safetensors version: 0.6.2
embedding-service       | - Accelerate version: not installed
embedding-service       | - Accelerate config: not found
embedding-service       | - DeepSpeed version: not installed
embedding-service       | - PyTorch version (accelerator?): 2.8.0+cu128 (NA)
embedding-service       | - Tensorflow version (GPU?): not installed (NA)
embedding-service       | - Flax version (CPU?/GPU?/TPU?): not installed (NA)
embedding-service       | - Jax version: not installed
embedding-service       | - JaxLib version: not installed
embedding-service       | - Using distributed or parallel set-up in script?: (I'm not sure tbh...)

### Who can help?

@ArthurZucker 

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

1. Use CPU+RAM
2. UPD: Add at least 1000 images to `data` folder
3. Git clone this repository: https://github.com/Adefey/search_dir, checkout to this commit: `d8a40baa434c4b00ad04dada9d7221edb111f4aa`
4. Run: docker compose up --build
5. Call API: POST http://localhost:8003/api/v1/start_discovery
6. Monitor logs, depending on available system memory embedding-service may get OOM (embedding-service exited with code 137) after some batches (on my system with 14 GB ram it's 10 batches)
But actually there are only 3 methods (`__init__`, `_encode`,  `encode_images`) that may be relevant. Repeated calls of `encode_images` result in OOM. I'll show relevant methods here:

```
def __init__(self):
        self.model_checkpoint = "openai/clip-vit-base-patch32"
        os.system("transformers env")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_checkpoint, use_fast=False)
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")
```

```
def _encode(self, inputs: dict) -> list[float]:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
[… 15 lines of code omitted]

```
def encode_images(self, images: list[bytes]) -> list[list[float]]:
        """
        Process images into embeddings
        """
[… 13 lines of code omitted]

Also, there is a working fix: calling `trim_memory()` after each model call:

```
def trim_memory():
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)
```

But I think this is a workaround and transformers library should manage resources correctly on its own.

### Expected behavior

Usecase: microservice with Model gets batches of 30 images to calculate embeddings, then it gets another batch. After 10 batches service is killed because of OOM. Manually monitoring memory in htop - usage increases with every batch by 600...800 MB. Expected behavior - constant memory usage for at every time during batch processing.
I suppose there is a memory leak or memory fragmentation issue where new memory keeps being allocated and not reused. 

UPD: exactly same OOM issue happens with `google/siglip2-base-patch16-256` and again `malloc_trim` workaround works

## [c3346651154] Rocketknight1 (MEMBER) · issue_comment · 2025-09-29
Hi, we can't run a reproducer that depends on checking out your entire repo and spinning up an entire docker image. Can you try to cut this down to a minimal reproducer (maybe using a Hub dataset so you can just download it in one line?) so we can just run the code in a notebook and see the issue?

## [c3348770452] Adefey (NONE) · issue_comment · reply to [c3346651154] · 2025-09-29
> Hi, we can't run a reproducer that depends on checking out your entire repo and spinning up an entire docker image. Can you try to cut this down to a minimal reproducer (maybe using a Hub dataset so you can just download it in one line?) so we can just run the code in a notebook and see the issue?

Hi,
I made a reproduction in Google Colab. It does not reproduce every time, however it reproduced consistently with with FastAPI/Uvicorn. This exact reproduction was done after running this notebook twice (without reset)
https://colab.research.google.com/drive/11xq99KQh1z72PVmPoECPEmT26ybZjirN?usp=sharing

Only model code runs there with and without `malloc_trim`. You can see that with it memory consumption is constant and without it slowly grows. In my scenario with FastAPI/Uvicorn it grown much faster for some reason.

Also there are some existing issues in Pytorch with same issue:
https://github.com/pytorch/pytorch/issues/68114
https://github.com/pytorch/pytorch/issues/114455

## [c3351739123] ydshieh (COLLABORATOR) · issue_comment · reply to [c3348770452] · 2025-09-30
@Adefey Could you check if #41215 fix the issue (especially with `FastAPI/Uvicorn`) 🙏

## [c3352967301] Adefey (NONE) · issue_comment · reply to [c3351739123] · 2025-09-30
Hi
@ydshieh
I checked it, but the issue remains with PR

Also I noticed that after manual image preprocessing (resize to 224x224) memory grows much slower (but still noticeable), almost unnoticeable. Without it (as was reported in this bug) memory grows much faster. My first approach without image preprocessing was relying in internal preprocessing done in AutoProcessor. As you can see in my reproducer in Colab - I feed processor with raw images without any preprocessing.
Is it possible that the issue in Processor and not Model?

## [c3355600824] ydshieh (COLLABORATOR) · issue_comment · reply to [c3352967301] · 2025-10-01
Thanks for double checking.

> I checked it, but the issue remains with PR

Is this (still) happening with your original reproducible (i.e. with FastAPI) or also on the Colab reproducer ?

> Is it possible that the issue in Processor and not Model?

Could you try to disable the part involving models (i.e. only keep the processor part) and see if we still have the memory issue?

## [c3367462006] Adefey (NONE) · issue_comment · reply to [c3355600824] · 2025-10-03
> Thanks for double checking.
> 
> > I checked it, but the issue remains with PR
> 
> Is this (still) happening with your original reproducible (i.e. with FastAPI) or also on the Colab reproducer ?
> 
> > Is it possible that the issue in Processor and not Model?
> 
> Could you try to disable the part involving models (i.e. only keep the processor part) and see if we still have the memory issue?

Yes, it still happens with FastAPI/uvicorn. I tried to reprodue it again (in the same colab) with processor only, but it did not reproduce. I'm not sure what exactly causes problem especially with FastAPI app.

## [c3411313752] ydshieh (COLLABORATOR) · issue_comment · 2025-10-16
I could try to dive into it, if you are able to a minimal example: ideally, a single simple python script (loading process / models, simple usage of FastAPI etc), an instruction of installing dependencies, i.e. don't include the whole things in your repo . I think it's doable despite it will take you more time to prepare the minimal example.

## [c3447749542] Adefey (NONE) · issue_comment · 2025-10-25
Hi

It seems it's a FastAPI/uvicorn issue after all. I wrote 2 scripts - model usage in loop and model usage in loop through http request to a FastAPI app. How can usage of FastAPI/uvicorn affect memory consumption?

[collapsed block 'Logs from script with FastAPI', 41 lines omitted]

[collapsed block 'Logs from script without FastAPI', 22 lines omitted]

Prepare environment:

`pip install -r requirements.txt`

Run without FastAPI:

`python main.py`

Run with FastAPI:

`python main_fastapi.py`

Repo with reproduction:
https://github.com/Adefey/transformers_leak_reproduction

## [c3451320666] ydshieh (COLLABORATOR) · issue_comment · reply to [c3447749542] · 2025-10-27
@Adefey I don't have the insight about it, but it seems it's a common issue with FastAPI if one searches with Google

https://github.com/fastapi/fastapi/discussions/11079
https://www.reddit.com/r/learnpython/comments/1669z84/unresolved_memory_management_issues_in/
https://www.google.com/search?q=fastapi+memory+leak&oq=fastapi+memory+leak&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIICAEQABgWGB4yCAgCEAAYFhgeMggIAxAAGBYYHjIICAQQABgWGB4yCAgFEAAYFhgeMggIBhAAGBYYHjIICAcQABgWGB4yCAgIEAAYFhgeMggICRAAGBYYHtIBCDQxOTNqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8

But it might require some time to read the discussions and try.

---
outcome: state=closed
