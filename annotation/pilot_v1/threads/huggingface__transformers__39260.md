# #39260 [issue] Please help i am trying to run model but issue
https://github.com/huggingface/transformers/issues/39260  ·  opened by vinaydanidhariya on 2025-07-07

## [body] vinaydanidhariya (NONE) · opening post · 2025-07-07
```
# install libraries
# Uninstall broken stable PyTorch
!pip uninstall -y torch torchvision torchaudio

# Install latest nightly PyTorch compatible with Python 3.12 (for CPU)
[… 31 lines of code omitted]
```
Found existing installation: torch 2.9.0.dev20250707+cpu
Uninstalling torch-2.9.0.dev20250707+cpu:
  Successfully uninstalled torch-2.9.0.dev20250707+cpu
Found existing installation: torchvision 0.24.0.dev20250707+cpu
Uninstalling torchvision-0.24.0.dev20250707+cpu:
  Successfully uninstalled torchvision-0.24.0.dev20250707+cpu
Found existing installation: torchaudio 2.8.0.dev20250707+cpu
Uninstalling torchaudio-2.8.0.dev20250707+cpu:
  Successfully uninstalled torchaudio-2.8.0.dev20250707+cpu
Looking in indexes: https://download.pytorch.org/whl/nightly/cpu
Collecting torch
  Using cached https://download.pytorch.org/whl/nightly/cpu/torch-2.9.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (28 kB)
Collecting torchvision
  Using cached https://download.pytorch.org/whl/nightly/cpu/torchvision-0.24.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (6.2 kB)
Collecting torchaudio
  Using cached https://download.pytorch.org/whl/nightly/cpu/torchaudio-2.8.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (7.3 kB)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.13.2)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.11/dist-packages (from torch) (1.14.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.4.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (1.26.4)
Requirement already satisfied: pillow!=8.3.*,<11.3,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.1.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)
Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (1.3.8)
Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (1.2.4)
Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (0.1.1)
Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (2025.1.0)
Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (2022.1.0)
Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy->torchvision) (2.4.1)
Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->torchvision) (2024.2.0)
Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy->torchvision) (2022.1.0)
Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy->torchvision) (1.3.0)
Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy->torchvision) (2024.2.0)
Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy->torchvision) (2024.2.0)
Using cached https://download.pytorch.org/whl/nightly/cpu/torch-2.9.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl (184.4 MB)
Using cached https://download.pytorch.org/whl/nightly/cpu/torchvision-0.24.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl (2.0 MB)
Using cached https://download.pytorch.org/whl/nightly/cpu/torchaudio-2.8.0.dev20250707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl (1.8 MB)
Installing collected packages: torch, torchaudio, torchvision
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
fastai 2.7.19 requires torch<2.7,>=1.10, but you have torch 2.9.0.dev20250707+cpu which is incompatible.
Successfully installed torch-2.9.0.dev20250707+cpu torchaudio-2.8.0.dev20250707+cpu torchvision-0.24.0.dev20250707+cpu
Requirement already satisfied: transformers==4.41.2 in /usr/local/lib/python3.11/dist-packages (4.41.2)
Requirement already satisfied: peft==0.11.1 in /usr/local/lib/python3.11/dist-packages (0.11.1)
Requirement already satisfied: librosa==0.10.2 in /usr/local/lib/python3.11/dist-packages (0.10.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (3.18.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (0.31.1)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (2024.11.6)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (2.32.3)
Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (0.19.1)
Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (0.5.3)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers==4.41.2) (4.67.1)
Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from peft==0.11.1) (7.0.0)
Requirement already satisfied: torch>=1.13.0 in /usr/local/lib/python3.11/dist-packages (from peft==0.11.1) (2.9.0.dev20250707+cpu)
Requirement already satisfied: accelerate>=0.21.0 in /usr/local/lib/python3.11/dist-packages (from peft==0.11.1) (1.5.2)
Requirement already satisfied: audioread>=2.1.9 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (3.0.1)
Requirement already satisfied: scipy>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (1.15.2)
Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (1.2.2)
Requirement already satisfied: joblib>=0.14 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (1.5.0)
Requirement already satisfied: decorator>=4.3.0 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (4.4.2)
Requirement already satisfied: numba>=0.51.0 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (0.60.0)
Requirement already satisfied: soundfile>=0.12.1 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (0.13.1)
Requirement already satisfied: pooch>=1.1 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (1.8.2)
Requirement already satisfied: soxr>=0.3.2 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (0.5.0.post1)
Requirement already satisfied: typing-extensions>=4.1.1 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (4.13.2)
Requirement already satisfied: lazy-loader>=0.1 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (0.4)
Requirement already satisfied: msgpack>=1.0 in /usr/local/lib/python3.11/dist-packages (from librosa==0.10.2) (1.1.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.23.0->transformers==4.41.2) (2025.3.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.23.0->transformers==4.41.2) (1.1.0)
Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /usr/local/lib/python3.11/dist-packages (from numba>=0.51.0->librosa==0.10.2) (0.43.0)
Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (1.3.8)
Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (1.2.4)
Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (0.1.1)
Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (2025.1.0)
Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (2022.1.0)
Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers==4.41.2) (2.4.1)
Requirement already satisfied: platformdirs>=2.5.0 in /usr/local/lib/python3.11/dist-packages (from pooch>=1.1->librosa==0.10.2) (4.3.8)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers==4.41.2) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers==4.41.2) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers==4.41.2) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers==4.41.2) (2025.4.26)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=0.20.0->librosa==0.10.2) (3.6.0)
Requirement already satisfied: cffi>=1.0 in /usr/local/lib/python3.11/dist-packages (from soundfile>=0.12.1->librosa==0.10.2) (1.17.1)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.11/dist-packages (from torch>=1.13.0->peft==0.11.1) (1.14.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=1.13.0->peft==0.11.1) (3.4.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.13.0->peft==0.11.1) (3.1.6)
Requirement already satisfied: pycparser in /usr/local/lib/python3.11/dist-packages (from cffi>=1.0->soundfile>=0.12.1->librosa==0.10.2) (2.22)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy>=1.13.3->torch>=1.13.0->peft==0.11.1) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch>=1.13.0->peft==0.11.1) (3.0.2)
Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.17->transformers==4.41.2) (2024.2.0)
Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.17->transformers==4.41.2) (2022.1.0)
Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy>=1.17->transformers==4.41.2) (1.3.0)
Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy>=1.17->transformers==4.41.2) (2024.2.0)
Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy>=1.17->transformers==4.41.2) (2024.2.0)
Requirement already satisfied: triton in /usr/local/lib/python3.11/dist-packages (2.2.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from triton) (3.18.0)
Requirement already satisfied: accelerate in /usr/local/lib/python3.11/dist-packages (1.5.2)
Collecting accelerate
  Downloading accelerate-1.8.1-py3-none-any.whl.metadata (19 kB)
Requirement already satisfied: numpy<3.0.0,>=1.17 in /usr/local/lib/python3.11/dist-packages (from accelerate) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from accelerate) (25.0)
Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from accelerate) (7.0.0)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.11/dist-packages (from accelerate) (6.0.2)
Requirement already satisfied: torch>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from accelerate) (2.9.0.dev20250707+cpu)
Requirement already satisfied: huggingface_hub>=0.21.0 in /usr/local/lib/python3.11/dist-packages (from accelerate) (0.31.1)
Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from accelerate) (0.5.3)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (3.18.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (2025.3.2)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (2.32.3)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (4.67.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (4.13.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from huggingface_hub>=0.21.0->accelerate) (1.1.0)
Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (1.3.8)
Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (1.2.4)
Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (0.1.1)
Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (2025.1.0)
Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (2022.1.0)
Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy<3.0.0,>=1.17->accelerate) (2.4.1)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.11/dist-packages (from torch>=2.0.0->accelerate) (1.14.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=2.0.0->accelerate) (3.4.2)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=2.0.0->accelerate) (3.1.6)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy>=1.13.3->torch>=2.0.0->accelerate) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch>=2.0.0->accelerate) (3.0.2)
Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0.0,>=1.17->accelerate) (2024.2.0)
Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy<3.0.0,>=1.17->accelerate) (2022.1.0)
Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy<3.0.0,>=1.17->accelerate) (1.3.0)
Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy<3.0.0,>=1.17->accelerate) (2024.2.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub>=0.21.0->accelerate) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub>=0.21.0->accelerate) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub>=0.21.0->accelerate) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface_hub>=0.21.0->accelerate) (2025.4.26)
Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy<3.0.0,>=1.17->accelerate) (2024.2.0)
Downloading accelerate-1.8.1-py3-none-any.whl (365 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 365.3/365.3 kB 6.4 MB/s eta 0:00:00a 0:00:01
Installing collected packages: accelerate
  Attempting uninstall: accelerate
    Found existing installation: accelerate 1.5.2
    Uninstalling accelerate-1.5.2:
      Successfully uninstalled accelerate-1.5.2
Successfully installed accelerate-1.8.1
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   1534         try:
-> 1535             return importlib.import_module("." + module_name, self.__name__)
   1536         except Exception as e:

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _load_unlocked(spec)

/usr/lib/python3.11/importlib/_bootstrap_external.py in exec_module(self, module)

/usr/lib/python3.11/importlib/_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)

/usr/local/lib/python3.11/dist-packages/transformers/pipelines/__init__.py in <module>
     46 )
---> 47 from .audio_classification import AudioClassificationPipeline
     48 from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

/usr/local/lib/python3.11/dist-packages/transformers/pipelines/audio_classification.py in <module>
     20 from ..utils import add_end_docstrings, is_torch_available, is_torchaudio_available, logging
---> 21 from .base import Pipeline, build_pipeline_init_args
     22 

/usr/local/lib/python3.11/dist-packages/transformers/pipelines/base.py in <module>
     33 from ..image_processing_utils import BaseImageProcessor
---> 34 from ..modelcard import ModelCard
     35 from ..models.auto.configuration_auto import AutoConfig

/usr/local/lib/python3.11/dist-packages/transformers/modelcard.py in <module>
     47 )
---> 48 from .training_args import ParallelMode
     49 from .utils import (

/usr/local/lib/python3.11/dist-packages/transformers/training_args.py in <module>
     72 if is_accelerate_available():
---> 73     from accelerate.state import AcceleratorState, PartialState
     74     from accelerate.utils import DistributedType

/usr/local/lib/python3.11/dist-packages/accelerate/__init__.py in <module>
     15 
---> 16 from .accelerator import Accelerator
     17 from .big_modeling import (

/usr/local/lib/python3.11/dist-packages/accelerate/accelerator.py in <module>
     35 
---> 36 from .checkpointing import load_accelerator_state, load_custom_state, save_accelerator_state, save_custom_state
     37 from .data_loader import DataLoaderDispatcher, prepare_data_loader, skip_first_batches

/usr/local/lib/python3.11/dist-packages/accelerate/checkpointing.py in <module>
     21 
---> 22 from .utils import (
     23     MODEL_NAME,

/usr/local/lib/python3.11/dist-packages/accelerate/utils/__init__.py in <module>
     34 )
---> 35 from .dataclasses import (
     36     AORecipeKwargs,

/usr/local/lib/python3.11/dist-packages/accelerate/utils/dataclasses.py in <module>
     33 
---> 34 from .constants import (
     35     BETA_TP_AVAILABLE_PYTORCH_VERSION,

ImportError: cannot import name 'FSDP2_PYTORCH_VERSION' from 'accelerate.utils.constants' (/usr/local/lib/python3.11/dist-packages/accelerate/utils/constants.py)

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_129/2705754419.py in <cell line: 0>()
     15 # load the model pipeline on gpu:0
     16 # pipe = transformers.pipeline(model='sarvamai/shuka_v1', trust_remote_code=True, device=0, torch_dtype='bfloat16')
---> 17 from transformers import pipeline
     18 
     19 pipe = pipeline(

/usr/lib/python3.11/importlib/_bootstrap.py in _handle_fromlist(module, fromlist, import_, recursive)

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   1523             value = self._get_module(name)
   1524         elif name in self._class_to_module.keys():
-> 1525             module = self._get_module(self._class_to_module[name])
   1526             value = getattr(module, name)
   1527         else:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   1535             return importlib.import_module("." + module_name, self.__name__)
   1536         except Exception as e:
-> 1537             raise RuntimeError(
   1538                 f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
   1539                 f" traceback):\n{e}"

RuntimeError: Failed to import transformers.pipelines because of the following error (look up to see its traceback):
cannot import name 'FSDP2_PYTORCH_VERSION' from 'accelerate.utils.constants' (/usr/local/lib/python3.11/dist-packages/accelerate/utils/constants.py)

## [c3071123378] ArkVex (CONTRIBUTOR) · issue_comment · 2025-07-14
What do you need help with broo...?
Can you please be a lil specific?

## [c3211537387] vinaydanidhariya (NONE) · issue_comment · 2025-08-21
I TRY TO RUN SURVAM MODEL BUT NOT WORKING CAN HELP

sarvamai/shuka_v1

---
outcome: state=closed
