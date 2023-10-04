<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://opensource.fb.com/support-ukraine"><img alt="Support Ukraine" src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" /></a>
  <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
  <a href="https://app.circleci.com/pipelines/github/facebookresearch/fairseq/"><img alt="CicleCI Status" src="https://circleci.com/gh/facebookresearch/fairseq.svg?style=shield" /></a>
</p>

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.
This repo is forked from Fairseq and includes changes to run models on Habana
devices.

We provide reference implementations of various sequence modeling papers:

<details><summary>List of implemented papers</summary><p>

* **Convolutional Neural Networks (CNN)**
  + [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  + [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  + [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  + [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  + [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* **LightConv and DynamicConv models**
  + [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* **Long Short-Term Memory (LSTM) networks**
  + Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
* **Transformer (self-attention) networks**
  + Attention Is All You Need (Vaswani et al., 2017)
  + [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  + [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  + [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/README.adaptive_inputs.md)
  + [Lexically constrained decoding with dynamic beam allocation (Post & Vilar, 2018)](examples/constrained_decoding/README.md)
  + [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](examples/truncated_bptt/README.md)
  + [Adaptive Attention Span in Transformers (Sukhbaatar et al., 2019)](examples/adaptive_span/README.md)
  + [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  + [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  + [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  + [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  + [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  + [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
  + [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
  + [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
  + [Generating Medical Reports from Patient-Doctor Conversations Using Sequence-to-Sequence Models (Enarvi et al., 2020)](examples/pointer_generator/README.md)
  + [Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)](examples/linformer/README.md)
  + [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
  + [Deep Transformers with Latent Depth (Li et al., 2020)](examples/latent_depth/README.md)
  + [Unsupervised Cross-lingual Representation Learning for Speech Recognition (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979)
  + [Self-training and Pre-training are Complementary for Speech Recognition (Xu et al., 2020)](https://arxiv.org/abs/2010.11430)
  + [Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training (Hsu, et al., 2021)](https://arxiv.org/abs/2104.01027)
  + [Unsupervised Speech Recognition (Baevski, et al., 2021)](https://arxiv.org/abs/2105.11084)
  + [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al., 2021)](https://arxiv.org/abs/2109.11680)
  + [VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding (Xu et. al., 2021)](https://arxiv.org/pdf/2109.14084.pdf)
  + [VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding (Xu et. al., 2021)](https://aclanthology.org/2021.findings-acl.370.pdf)
  + [NormFormer: Improved Transformer Pretraining with Extra Normalization (Shleifer et. al, 2021)](examples/normformer/README.md)
* **Non-autoregressive Transformers**
  + Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  + Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  + Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  + Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  + [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* **Finetuning**
  + [Better Fine-Tuning by Reducing Representational Collapse (Aghajanyan et al. 2020)](examples/rxf/README.md)

</p></details>

# Requirements and Installation

* Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment. To achieve the best performance, please follow the methods outlined in the
[Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html). 
The guides will walk you through the process of setting up your system to run the model on Gaudi.
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/HabanaAI/fairseq
cd fairseq
pip install --editable ./
```

# Getting Started

List of models for which training has been tested on Habana devices:
  - [Wave2Vec 2.0](examples/wav2vec/README.md)
  - [Transformer](examples/translation/README.md)

In order to train another model available in fairseq (other than those listed above) on Habana device, please follow the instructions below,
  - Use "--hpu" argument when invoking command-line tools such as fairseq-train, fairseq-interactive, fairseq-generate etc.
  - Enable mixed precision training by using "--hpu-mixed-precision-mode autocast" when invoking command-line tools such as fairseq-train.
# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
