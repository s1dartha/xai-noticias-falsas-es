---
language:
- es
license: apache-2.0
multilinguality:
- monolingual
size_categories:
- n<1K
source_datasets:
- original
task_categories:
- summarization
pretty_name: NoticIA
dataset_info:
  features:
  - name: web_url
    dtype: string
  - name: web_headline
    dtype: string
  - name: summary
    dtype: string
  - name: web_text
    dtype: string
  splits:
  - name: train
    num_bytes: 2494253
    num_examples: 700
  - name: validation
    num_bytes: 214922
    num_examples: 50
  - name: test
    num_bytes: 358972
    num_examples: 100
  download_size: 1745629
  dataset_size: 3068147
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
tags:
- summarization
- clickbait
- news
---

<p align="center">
    <img src="https://huggingface.co/datasets/Iker/NoticIA/resolve/main/assets/logo.png" style="height: 250px;">
</p>

<h3 align="center">"A Clickbait Article Summarization Dataset in Spanish."</h3>

We present NoticIA, a dataset consisting of 850 Spanish news articles featuring prominent clickbait headlines, each paired with high-quality, single-sentence generative summarizations written by humans.

- 📖 Paper: [NoticIA: A Clickbait Article Summarization Dataset in Spanish](https://arxiv.org/abs/2404.07611)
- 💻 Baseline Code: [https://github.com/ikergarcia1996/NoticIA](https://github.com/ikergarcia1996/NoticIA)
- 🤖 Pre Trained Models [https://huggingface.co/collections/Iker/noticia-and-clickbaitfighter-65fdb2f80c34d7c063d3e48e](https://huggingface.co/collections/Iker/noticia-and-clickbaitfighter-65fdb2f80c34d7c063d3e48e)
- 🔌 Online Demo: [https://iker-clickbaitfighter.hf.space/](https://iker-clickbaitfighter.hf.space/)


For example, given the following headline and web text:
```
# ¿Qué pasará el 15 de enero de 2024?
Al parecer, no todo es dulzura en las vacaciones de fin de años, como lo demuestra la nueva intrig....
```
The summary is:
```
Que los estudiantes vuelven a clase.
```


# Data explanation
- **web_url** (int): The URL of the news article
- **web_headline** (str):  The headline of the article, which is a Clickbait. 
- **web_text** (int):  The body of the article. 
- **summary** (str):  The summary written by humans that answers the clickbait headline. 

# Dataset Description
- **Author:** [Iker García-Ferrero](https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/)
- **Author** [Begoña Altuna](https://www.linkedin.com/in/bego%C3%B1a-altuna-78014139)
- **Web Page**: [Github](https://github.com/ikergarcia1996/NoticIA)
- **Language(s) (NLP):** Spanish
- **License:** apache-2.0

# Dataset Usage

1. We are working on implementing NoticIA on the Language Model Evaluation Harness library: https://github.com/EleutherAI/lm-evaluation-harness

2. If you want to train an LLM or reproduce the results in our paper, you can use our code. See the repository for more info: [https://github.com/ikergarcia1996/NoticIA](https://github.com/ikergarcia1996/NoticIA)

3. If you want to manually load the dataset and run inference with an LLM: 
You can load the dataset with the following command:
```Python
from datasets import load_dataset
dataset = load_dataset("Iker/NoticIA")
```

In order to perform inference with LLMs, you need to build a prompt. The one we use in our paper is:
```Python
def clickbait_prompt(
    headline: str,
    body: str,
) -> str:
    """
    Generate the prompt for the model.
    Args:
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
    Returns:
        `str`: The formatted prompt.
    """
    return (
        f"Ahora eres una Inteligencia Artificial experta en desmontar titulares sensacionalistas o clickbait. "
        f"Tu tarea consiste en analizar noticias con titulares sensacionalistas y "
        f"generar un resumen de una sola frase que revele la verdad detrás del titular.\n"
        f"Este es el titular de la noticia: {headline}\n"
        f"El titular plantea una pregunta o proporciona información incompleta. "
        f"Debes buscar en el cuerpo de la noticia una frase que responda lo que se sugiere en el título. "
        f"Responde siempre que puedas parafraseando el texto original. "
        f"Usa siempre las mínimas palabras posibles. "
        f"Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n"
        f"{body}\n"
    )
```

Here is a practical end-to-end example using the text generation pipeline.
```python
from transformers import pipeline
from datasets import load_dataset

generator = pipeline(model="google/gemma-2b-it",device_map="auto")
dataset = load_dataset("Iker/NoticIA")

example = dataset["test"][0]
prompt = clickbait_prompt(headline=example["web_headline"],body=example["web_text"])
outputs = generator(prompt, return_full_text=False,max_length=4096)
print(outputs)

# [{'generated_text': 'La tuitera ha recibido un número considerable de comentarios y mensajes de apoyo.'}]
```

Here is a practical end-to-end example using the generate function
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",device_map="auto",quantization_config={"load_in_4bit": True})
dataset = load_dataset("Iker/NoticIA")

example = dataset["test"][0]
prompt = clickbait_prompt(headline=example["web_headline"],body=example["web_text"])
prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
model_inputs = tokenizer(
        text=prompt,
        max_length=3096,
        truncation=True,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )

outputs = model.generate(**model_inputs,max_length=4096)
output_text = tokenizer.batch_decode(outputs)

print(output_text[0])

# La usuaria ha comprado un abrigo para su abuela de 97 años, pero la "yaya" no está de acuerdo.
```


# Uses
This dataset is intended to build models tailored for academic research that can extract information from large texts. The objective is to research whether current LLMs, given a question formulated as a Clickbait headline, can locate the answer within the article body and summarize the information in a few words. The dataset also aims to serve as a task to evaluate the performance of current LLMs in Spanish.


# Out-of-Scope Use
You cannot use this dataset to develop systems that directly harm the newspapers included in the dataset. This includes using the dataset to train profit-oriented LLMs capable of generating articles from a short text or headline, as well as developing profit-oriented bots that automatically summarize articles without the permission of the article's owner. Additionally, you are not permitted to train a system with this dataset that generates clickbait headlines.

This dataset contains text and headlines from newspapers; therefore, you cannot use it for commercial purposes unless you have the license for the data.


# Dataset Creation
The dataset has been meticulously created by hand. We utilize two sources to compile Clickbait articles:
- The Twitter user [@ahorrandoclick1](https://twitter.com/ahorrandoclick1), who reposts Clickbait articles along with a hand-crafted summary. Although we use their summaries as a reference, most of them have been rewritten (750 examples from this source).
- The web demo [⚔️ClickbaitFighter⚔️](https://iker-clickbaitfighter.hf.space/), which operates a pre-trained model using an early iteration of our dataset. We collect all the model inputs/outputs and manually correct them (100 examples from this source).

# Who are the annotators?
The dataset was annotated by [Iker García-Ferrero](https://ikergarcia1996.github.io/Iker-Garcia-Ferrero/) and validated by [Begoña Altuna](https://www.linkedin.com/in/bego%C3%B1a-altuna-78014139).
The annotation took ~40 hours.

# Citation

```bittext
@misc{noticia2024,
      title={NoticIA: A Clickbait Article Summarization Dataset in Spanish}, 
      author={Iker García-Ferrero and Begoña Altuna},
      year={2024},
      eprint={2404.07611},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```