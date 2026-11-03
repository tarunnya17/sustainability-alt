# #38657 [issue] please develop transformers java/scala  sdk,eagerly to use!
https://github.com/huggingface/transformers/issues/38657  ·  opened by mullerhai on 2025-06-07

## [body] mullerhai (NONE) · opening post · 2025-06-07
### Feature request

Transformers have become de facto essential deep learning tools. However, you only provide interface APIs for three languages: Python, JavaScript, and Rust. Have you forgotten about Java, Scala, and Kotlin? Is it because your team lacks manpower or isn't proficient in the JVM? Currently, there are also PyTorch bindings for Java and Scala, such as javacpp - pytorch and storch. Everyone wants to use Transformers, yet you haven't overcome this obstacle for so many years. Please tell your technical director or CTO that we need Java and Scala libraries for Transformers, which can directly download model datasets and Tokenizers and perform model training. Please make sure to solve this issue this year. Thank you

### Motivation

dd

### Your contribution

dd

## [c2956927797] AshAnand34 (CONTRIBUTOR) · issue_comment · reply to [body] · 2025-06-09
Hi @mullerhai,

I'm a new contributor to the Hugging Face community, and I came across your issue while exploring the codebase. I understand your frustration about the lack of Java/Scala support in Transformers. You're absolutely right that JVM languages are widely used in enterprise environments, and having native support would be valuable for many users. As someone who's just starting to contribute, I'd like to help move this forward and learn from your experience.

I see you've mentioned existing PyTorch bindings like javacpp-pytorch and storch. To help us prioritize and implement this feature effectively, could you share:

1. In your current setup:
   - Are you using any of these existing bindings (javacpp-pytorch, storch) in production?
   - What specific limitations are you encountering with the current solutions?

2. For your use case:
   - Which specific models are you most interested in using?
   - Do you need both training and inference capabilities?
   - Are you working with any specific JVM-based ML frameworks that we should ensure compatibility with?

3. Regarding deployment:
   - Are you running this in a production environment?
   - What are your current performance requirements (throughput, latency, etc.)?

As a new community member, I'm particularly interested in understanding the real-world needs of JVM users like yourself. Your experience with the current limitations would be invaluable in helping me and others shape this implementation effectively.

Thank you for your patience and for helping me learn more about the community's needs!

## [c2957412726] mullerhai (NONE) · issue_comment · reply to [c2956927797] · 2025-06-10
> Hi [@mullerhai](https://github.com/mullerhai),
> 
> I'm a new contributor to the Hugging Face community, and I came across your issue while exploring the codebase. I understand your frustration about the lack of Java/Scala support in Transformers. You're absolutely right that JVM languages are widely used in enterprise environments, and having native support would be valuable for many users. As someone who's just starting to contribute, I'd like to help move this forward and learn from your experience.
> 
> I see you've mentioned existing PyTorch bindings like javacpp-pytorch and storch. To help us prioritize and implement this feature effectively, could you share:
> 
> 1. In your current setup:
>    
>    * Are you using any of these existing bindings (javacpp-pytorch, storch) in production?
>    * What specific limitations are you encountering with the current solutions?
> 2. For your use case:
>    
>    * Which specific models are you most interested in using?
>    * Do you need both training and inference capabilities?
>    * Are you working with any specific JVM-based ML frameworks that we should ensure compatibility with?
> 3. Regarding deployment:
>    
>    * Are you running this in a production environment?
>    * What are your current performance requirements (throughput, latency, etc.)?
> 
> As a new community member, I'm particularly interested in understanding the real-world needs of JVM users like yourself. Your experience with the current limitations would be invaluable in helping me and others shape this implementation effectively.
> 
> Thank you for your patience and for helping me learn more about the community's needs!

very tnanks , it is excited!  please make it realable for transformers  in jvm scala3!
1. usage

I am already using Storch in production. You can check my repository storch-recommend. Our recommendation system has started using Storch.

Storch relies on JavaCPP Presets for PyTorch. Let me briefly introduce it: JavaCPP Presets for PyTorch is a complete mapping of C++ LibTorch, generating glue Java code similar to JNA/JNI. It provides low-level APIs, which are cumbersome to use and lack documentation. If you haven't used it, you may encounter unexpected failures due to a lack of relevant knowledge, especially when configuring hyperparameters for some network layers. There are multiple ways to set values, and a wrong setting will prevent effective value transmission. We encountered this issue during the development of Storch and wasted a lot of time. We also found bugs in the hyperparameter configuration of several padding network layers, but JavaCPP hasn't fixed them. However, they provide most of the effective tensor operators and PyTorch methods, which we have encapsulated in Storch. In the future, I also suggest you communicate with the JavaCPP official responsible person, Sadnet. They plan to release bindings based on LibTorch 2.7, but they are currently encountering compilation issues with CUDA 12.9 in JavaCPP. Yes, they even use JavaCPP to generate JNI bindings for CUDA. Additionally, JavaCPP Presets for PyTorch provides dataset, DataLoader, and sampler, which we have also encapsulated. We found during testing that if the system's GCC and clib versions are too low, it will report errors. Generally, a higher-version Ubuntu system is used. Another important point is that JavaCPP's distributed training currently only supports Gloo, while CUDA for the standalone version is supported. I believe this is an issue that needs to be addressed in the future, and we need to communicate with JavaCPP together. JavaCPP refers to the Java binding of LibTorch. Some operators and network layers in PyTorch are not available in LibTorch, and therefore not in JavaCPP. But if we know the principles, we can implement them ourselves.

Storch is a secondary development of LibTorch bindings for Scala 3 based on JavaCPP Presets for PyTorch. Storch provides more user-friendly and flexible APIs, combining the type system and functional programming to make development closer to Python. Even Python code can be directly used in Storch with just a few lines of code modified. You can refer to the code in my storch-tutorial repository, which is very beneficial for migrating from Python to Scala 3 Storch. Storch can provide an enterprise-level model inference platform and train faster than Python. Additionally, the PT model files trained by Python PyTorch cannot be directly loaded in Storch. They need to be converted to TorchScript using torch.jit.trace. I think this is quite inconvenient. It seems that using the PT model in C++ LibTorch also requires conversion to TorchScript.

There are many advantages of using Scala 3. Java requires writing a lot of cumbersome boilerplate code. It is more suitable to prioritize the development of transformers in the Scala 3 version. Now, there are a large number of Scala 3 developers in Germany, the United States, and India. In the future, only simple Java API bindings need to be generated.

Additionally, note that Scala 3 should be used preferably, and Scala 2.12 and Scala 2.13 should be avoided as much as possible.

Limitations Encountered

We are developing directly using transformers with a large number of pre-trained speech large models, NLP LLMs, and machine vision models such as YOLOv11, along with their tokenizers and image processing programs, for deep training or direct application inference and prediction. However, we currently lack these resources. I have just implemented how to download large model files from Hugging Face Transformers and Safetensors GGUF format for Scala 3 development. There are also Scala 3 versions of NumPy, Pickle, and Polars, which you can use directly in the future. Now, Storch has the capability to handle subsequent operations.

Additionally, we need Hugging Face Datasets,TRL reinforcement learning library, and model quantization and optimization libraries.

2. Models of Interest

Hugging Face Transformers officially provides nearly 100 models. I believe we should try to cover every type of model and need various tokenizers and image processing processors.

NLP & LLM

DeepSeek, GPT, Qianwen, Gemmi, BERT

Computer Vision

YOLOv11, YOLOv12, U-Net, Mistral, Mask2Former

Multimodal

SmolVLM, FastLVM, SmolDocling, CLIP

I need both training and inference capabilities, with priority given to inference applications.

Other JVM ML frameworks, such as TensorFlow, DJL, EasyAI, and MXNet, are no longer in use because they are not user-friendly. PyTorch has become the standard for deep learning, and other frameworks have many disadvantages compared to it. Now, it is unnecessary to invest effort in compatibility with them. I personally believe that promoting the prosperity of LibTorch on the JVM will make it easier to unify multiple language frameworks, and they will follow suit.

Deployment
Directly use the java -jar command-line mode to load the model.
Use Docker containers or K8s containers to load the model program.
Use TorchServe for deployment.
Develop frameworks similar to VLLM, SGLang, and DeepSpeed in the future for deployment.
Performance Requirements

The performance should be able to handle 1k-10k QPS of concurrent access in the future, with a latency of 300ms-2500ms.

Additionally, if you have any questions in the future, you can contact me or find some projects you can use from my repositories for dependencies. You can also search for repositories starting with "storch" in the Maven Central Repository. 
There are three other original authors of Storch, who are very skilled programmers. They will give you a lot of inspiration. @sbrunk 
https://github.com/sbrunk

Also, join the Scala community on Discord. Many people are looking forward to your collaboration and participation. Now, the entire Scala 3 community is seeking breakthroughs in deep learning with Scala, as Scala has lagged behind compared to Python, Rust, and Mojo.

Link to Scala Community Discussion
https://contributors.scala-lang.org/t/compiling-scala-to-python-as-a-new-platform-for-data-engineering-and-ai/7129/32

https://github.com/mullerhai/storch
https://github.com/mullerhai/storch-safe-tensor
https://github.com/mullerhai/storch-numpy
https://github.com/mullerhai/storch-pickle
https://github.com/mullerhai/storch-recommend
https://github.com/mullerhai/storch-tutorial

## [c2957538903] mullerhai (NONE) · issue_comment · reply to [c2957412726] · 2025-06-10
> Hi [@mullerhai](https://github.com/mullerhai),
> 
> I'm a new contributor to the Hugging Face community, and I came across your issue while exploring the codebase. I understand your frustration about the lack of Java/Scala support in Transformers. You're absolutely right that JVM languages are widely used in enterprise environments, and having native support would be valuable for many users. As someone who's just starting to contribute, I'd like to help move this forward and learn from your experience.
> 
> I see you've mentioned existing PyTorch bindings like javacpp-pytorch and storch. To help us prioritize and implement this feature effectively, could you share:
> 
> 1. In your current setup:
>    
>    * Are you using any of these existing bindings (javacpp-pytorch, storch) in production?
>    * What specific limitations are you encountering with the current solutions?
> 2. For your use case:
>    
>    * Which specific models are you most interested in using?
>    * Do you need both training and inference capabilities?
>    * Are you working with any specific JVM-based ML frameworks that we should ensure compatibility with?
> 3. Regarding deployment:
>    
>    * Are you running this in a production environment?
>    * What are your current performance requirements (throughput, latency, etc.)?
> 
> As a new community member, I'm particularly interested in understanding the real-world needs of JVM users like yourself. Your experience with the current limitations would be invaluable in helping me and others shape this implementation effectively.
> 
> Thank you for your patience and for helping me learn more about the community's needs!

by the way , if you need , maybe you could  connect me  on discord ,and email  hai710459649@gmail.com  ,wechat mullerhelen

---
outcome: state=open
