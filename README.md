# ViT5
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vit5-pretrained-text-to-text-transformer-for/abstractive-text-summarization-on-vietnews)](https://paperswithcode.com/sota/abstractive-text-summarization-on-vietnews?p=vit5-pretrained-text-to-text-transformer-for)


A pretrained Transformer-based encoder-decoder model for the Vietnamese language. With [T5](https://github.com/google-research/text-to-text-transfer-transformer)-style self-supervised pretraining, ViT5 is trained on a large corpus of high-quality and diverse Vietnamese texts. We benchmark ViT5 on two downstream text generation tasks, Abstractive Text Summarization and Named Entity Recognition. All the experiments are shown in our paper [ViT5: Pretrained Text-to-Text Transformer for Vietnamese Language Generation](https://arxiv.org/abs/2205.06457)

### News Summarization Demo
Try our demo on [HF Spaces](https://huggingface.co/spaces/VietAI/ViNewsSum)

### 🤗 HuggingFace Model Checkpoint
- [ViT5-Base-1024 (1M)](https://huggingface.co/VietAI/vit5-base)
- [ViT5-Large-1024 (1.5M)](https://huggingface.co/VietAI/vit5-large)

_Notes: These two checkpoints are just pretrained language models._

#### Example
Below, we give an example of how to load ViT5 model from HuggingFace to summarize documents:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model.to("cuda")

sentence = "VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam."
text =  "vietnews: " + sentence + " </s>"
encoding = tokenizer(text, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    early_stopping=True
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)
```

Load our pretrained models on HuggingFace

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Base
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")

# Large
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large")  
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large")
```

## Evaluation

- [Eval scripts for Vietnews Summarization with HuggingFace 🤗](https://github.com/vietai/ViT5/blob/main/eval/Eval_vietnews_sum.ipynb)

![image](https://user-images.githubusercontent.com/44376091/187878636-15310ddb-7065-456d-8276-e606df482087.png)



### Datasets
- [Wikilingua](https://github.com/esdurmus/Wikilingua)
- [Vietnews](https://github.com/ThanhChinhBK/vietnews)
- [Pho_NER](https://github.com/VinAIResearch/PhoNER_COVID19)


### Finetuning
#### Abstractive Text Summarization
For easily reproducing our results, we provide the ViT5 checkpoint finetuned on vietnews as well. You can directly use our model on [HuggingFace](https://huggingface.co/VietAI/vit5-large-vietnews-summarization) 🤗.


## Citation
```
@inproceedings{phan-etal-2022-vit5,
    title = "{V}i{T}5: Pretrained Text-to-Text Transformer for {V}ietnamese Language Generation",
    author = "Phan, Long and Tran, Hieu and Nguyen, Hieu and Trinh, Trieu H.",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Student Research Workshop",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-srw.18",
    pages = "136--142",
}
```

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
We would like to thank Google for the support of Cloud credits and TPU quota!
