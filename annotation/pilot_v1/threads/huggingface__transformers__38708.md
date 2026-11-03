# #38708 [issue] Bert2D: A 2D-Word Embedding Model for Morphologically Rich Languages
https://github.com/huggingface/transformers/issues/38708  ·  opened by yigit353 on 2025-06-09

## [body] yigit353 (NONE) · opening post · 2025-06-09
### Model description

**Bert2D** is a novel transformer-based model that builds upon the `BertModel` architecture by introducing a two-dimensional word embedding system. This enhancement is specifically designed to improve performance on morphologically rich languages, such as Turkish, Finnish, and Hungarian. This model card describes the initial release, which includes the model implementation and a pretrained checkpoint for Turkish.

This work is based on the research outlined in the paper **"Bert2D: A 2D-Word Embedding for Morphologically Rich Languages"**, which has been accepted by IEEE and is available at: [https://ieeexplore.ieee.org/document/10542953](https://ieeexplore.ieee.org/document/10542953).

A pretrained model for Turkish, `Bert2D-cased-Turkish-128K-WWM-NSW2`, is available on the Hugging Face Hub at: [https://huggingface.co/yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2](https://huggingface.co/yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2)

## Model Description

The primary innovation of **Bert2D** is its use of a 2D positional embedding mechanism to better capture the complex morphological structures present in agglutinative languages. Unlike standard BERT models that use a 1D positional embedding, Bert2D employs a dual system:

1.  **Whole-Word Positional Embeddings (1st Dimension):** This captures the absolute position of each word in a sequence.
2.  **Sub-word Relative Positional Embeddings (2nd Dimension):** This encodes the relative position of sub-words within each word, allowing the model to distinguish between the beginning, middle, and end of a word's sub-tokens.

This two-dimensional approach provides a more nuanced representation of meaning by enabling the model to understand the relationships between words and their constituent morphemes. The model also incorporates **Whole Word Masking (WWM)**, a training technique where all sub-tokens corresponding to a single word are masked, encouraging the model to learn deeper contextual relationships.

### Architectural Innovations

The key components introduced in this release are:

* **`Bert2DModel`**: A new model class inheriting from `BertPreTrainedModel` that implements the 2D embedding logic. The core modifications are in the embeddings layer to accommodate the dual positional encoding.
* **`Bert2DTokenizer` and `Bert2DTokenizerFast`**: Custom tokenizers compatible with the `Bert2D` model.
* **Model Variants**: Includes all standard BERT architecture variants, such as:
    * `Bert2DForMaskedLM`
    * `Bert2DForSequenceClassification`
    * `Bert2DForTokenClassification`
    * `Bert2DForQuestionAnswering`

### Configuration Parameters

The `Bert2DConfig` includes new parameters to manage the 2D embeddings:

* `max_word_position_embeddings`: Defines the maximum number of words (not sub-tokens) the model can handle in a sequence. The default is `512`.
* `max_intermediate_subword_position_embeddings`: Specifies the embedding value for intermediate sub-tokens within a word. For the `NSW2` strategy, this is set to `2`.

The 2D embeddings are summed with the token and segment embeddings before being passed to the Transformer layers. The parameter count is nearly identical to a standard BERT model; the `128K` in the checkpoint name refers to the vocabulary size.

### Example Usage

The `Bert2D` model can be easily used with the `pipeline` API for tasks such as `fill-mask`.

```python
from transformers import pipeline

# Initialize the fill-mask pipeline with the Bert2D model
fill_mask_pipe = pipeline("fill-mask", model="yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2")
[… 11 lines of code omitted]

**Predicted Output:**
```
Token: mühendis
Sequence: Adamın mesleği mühendis midir acaba?
Score: 0.2393
--------------------
Token: doktor
[… 15 lines of code omitted]
```

### Fine-Tuning Considerations

When fine-tuning a `Bert2D` model, it is crucial to use the model's specific configuration. The introduction of `max_word_position_embeddings` and `max_intermediate_subword_position_embeddings` means that standard BERT configuration files are not directly compatible. Ensure that you are using the `Bert2DConfig` and its associated parameters for optimal performance.

### Motivation and Context

Traditional NLP models often struggle with languages that have rich morphology, as the vast number of word forms for a single root makes it difficult for models with 1D positional embeddings to generalize effectively. The **Bert2D** architecture was developed to address this limitation, and initial experiments on Turkish have shown that it consistently outperforms strong monolingual models across a range of downstream tasks.

### Future Work and Contributions

The developers are actively seeking contributions in the following areas:

* **Pretraining on other languages:** Particularly other morphologically complex languages like Finnish, Hungarian, and Korean.
* **Further architectural enhancements.**
* **Downstream task fine-tuning and evaluation.**

### Open source status

- [x] The model implementation is available
- [x] The model weights are available

### Provide useful links for the implementation

Weights: https://huggingface.co/yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2
Article: [https://ieeexplore.ieee.org/document/10542953](https://ieeexplore.ieee.org/document/10542953)
The PR: #38707

---
outcome: state=open
