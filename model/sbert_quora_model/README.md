---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:363913
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: i have been selected in tech mahindra i havent given my confirmation
    till its deadline nor have i solved its modules will i get a call letter
  sentences:
  - how can i lose post marriage weight
  - i have 105 years of experience in it i have been given p1 level in tech mahindra
    is it the correct level for my experience
  - what are the best things about google
- source_sentence: lead and copper samples are collected from taps that have sat unused
    for at least six hours what are these samples called
  sentences:
  - can more people become millionaires or billionaires
  - are there any presentation methods that can show both the exact quantity of a
    sample group and the proportion of it to the entire sample size at once
  - what are your views on banning 500 and 1000 rupee notes how does it affect black
    money and is it really gonna work and expose all the black money
- source_sentence: why is quora so leftleaning and liberal
  sentences:
  - how do you know if you are in love with your crush
  - which religion is oldest religion in the earth
  - why is quora so leftwing
- source_sentence: how do i join any club at mit manipal
  sentences:
  - how do i start a club at mit manipal
  - is there any scientific proof for the existence of aliens
  - how should i prepare myself for campus placements
- source_sentence: can you tell if someone is slow
  sentences:
  - how do i tell someone what i am good at
  - in how many days one can learn python provided that he knows java and c language
    well
  - do ghosts really exists
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision 12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0 -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'can you tell if someone is slow',
    'how do i tell someone what i am good at',
    'in how many days one can learn python provided that he knows java and c language well',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 363,913 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 14.04 tokens</li><li>max: 40 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 14.34 tokens</li><li>max: 66 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.39</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                             | sentence_1                                                                                                  | label            |
  |:-----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what should i get my 16 year old brother for his birthday</code> | <code>what should i get my 17 almost 18 year old sister for her birthday</code>                             | <code>0.0</code> |
  | <code>what should i do to learn hacking</code>                         | <code>what is the best way to learn white hat hacking</code>                                                | <code>0.0</code> |
  | <code>hows is working environment in price water house coopers</code>  | <code>what competitions currently have the largest prize money for an individual or very small group</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0440 | 500   | 0.206         |
| 0.0879 | 1000  | 0.1596        |
| 0.1319 | 1500  | 0.1385        |
| 0.1759 | 2000  | 0.1321        |
| 0.2198 | 2500  | 0.1259        |
| 0.2638 | 3000  | 0.1228        |
| 0.3077 | 3500  | 0.1187        |
| 0.3517 | 4000  | 0.1165        |
| 0.3957 | 4500  | 0.1134        |
| 0.4396 | 5000  | 0.1116        |
| 0.4836 | 5500  | 0.1104        |
| 0.5276 | 6000  | 0.1095        |
| 0.5715 | 6500  | 0.1086        |
| 0.6155 | 7000  | 0.1074        |
| 0.6595 | 7500  | 0.1055        |
| 0.7034 | 8000  | 0.105         |
| 0.7474 | 8500  | 0.1022        |
| 0.7913 | 9000  | 0.1034        |
| 0.8353 | 9500  | 0.1033        |
| 0.8793 | 10000 | 0.099         |
| 0.9232 | 10500 | 0.101         |
| 0.9672 | 11000 | 0.1012        |
| 0.0440 | 500   | 0.0984        |
| 0.0879 | 1000  | 0.0948        |
| 0.1319 | 1500  | 0.0943        |
| 0.1759 | 2000  | 0.0921        |
| 0.2198 | 2500  | 0.0883        |
| 0.2638 | 3000  | 0.086         |
| 0.3077 | 3500  | 0.0829        |
| 0.3517 | 4000  | 0.0799        |
| 0.3957 | 4500  | 0.0761        |
| 0.4396 | 5000  | 0.0733        |
| 0.4836 | 5500  | 0.0693        |
| 0.5276 | 6000  | 0.0646        |
| 0.5715 | 6500  | 0.0618        |
| 0.6155 | 7000  | 0.0579        |
| 0.6595 | 7500  | 0.0562        |
| 0.7034 | 8000  | 0.0532        |
| 0.7474 | 8500  | 0.0497        |
| 0.7913 | 9000  | 0.0492        |
| 0.8353 | 9500  | 0.048         |
| 0.8793 | 10000 | 0.0445        |
| 0.9232 | 10500 | 0.0445        |
| 0.9672 | 11000 | 0.0428        |


### Framework Versions
- Python: 3.11.13
- Sentence Transformers: 4.1.0
- Transformers: 4.53.0
- PyTorch: 2.6.0+cu124
- Accelerate: 1.8.1
- Datasets: 2.14.4
- Tokenizers: 0.21.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->