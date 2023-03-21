# Wav2vec 2.0

# Table of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview
The base model used is Wav2vec 2.0. The model learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477).

The paper shows that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.
## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.
To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/fairseq
```

**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below.

```
export PYTHONPATH=/path/to/fairseq:$PYTHONPATH
```

### Install Model Requirements
In the docker container, go to Fairseq directory and install fairseq along with the required packages using pip:
```bash
cd fairseq
pip install --editable .
```

### Set up the Dataset
Follow the steps below to set up Wav2vec dataset:
1. Download the dataset from http://www.openslr.org/12.
2. Create the train-960 directory comprised of the untared train-clean-100, train-clean-360 train-other-500 ( totaling 960 hours of speech).
3. Run the following command to create the manifest file:
```bash
$PYTHON wav2vec_manifest.py /path/to/dataset/train-960/ --dest /path/to/dataset/train-960/manifest --valid-percent 0.05
```
You can obtain “wav2vec_manifest.py” file from /path/to/fairseq/examples/wav2vec.

An example layout of the dataset will look like below:
```
100/
1001/
1006/
101/
1012/
...
manifest/
```

Note:
1. Please make sure the first line in /path/to/dataset/train-960/manifest/train.tsv and /path/to/dataset/train-960/manifest/valid.tsv points to the correct directory. e.g. `/data/pytorch/wav2vec/data/LibriSpeech/train-960`.
2. Going forward we assume the above Wav2vec dataset is available at path `/data/pytorch/wav2vec/data/LibriSpeech/train-960`.

## Training and Examples
### Single Card Training
**Run training on 1 HPU:**
- Run training on 1 HPU, Gradient accumulation=64, mixed precision (BF16):
```bash
fairseq-hydra-train task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```

### Multi-Card Training
To run multi-card demo, the following is required:
- The host machine has 512 GB of RAM installed.
- Make sure to follow the [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install) to install and set up the docker, 
  so that it has access to all 8 cards required for multi-card demo.
- Before executing the multi-card demo scripts, make sure all server network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
  ```
  sudo ip link set <interface_name> up
  ```
  To identify if a specific network interface is managed by the habanalabs driver type, run:
  ```
  sudo ethtool -i <interface_name>
  ```

**Run training on 8 HPUs:**

**Note:** The number of cards can be configured using `--world_size` option in the demo script as shown below.

1. Modify the `wav2vec2_base_librispeech_hpu.yaml` under `/path/to/fairseq/examples/wav2vec/config/pretraining/`.

2. Set `distributed_world_size` to 8:
```
distributed_training:
  distributed_world_size: 8
```
3. Set `update_freq` to 8:
```
optimization:
  max_update: 400000
  lr: [0.0005]
  update_freq: [8]
```
4. Run the following command (first-gen Gaudi):
```bash
fairseq-hydra-train task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
5. Run the following command (Gaudi2):
```bash
PT_RECIPE_CACHE_PATH="./cache_dir/" common.log_interval=111 common.hpu_graphs=true fairseq-hydra-train task.data=/data/pytorch/wav2vec/data/LibriSpeech/train-960/manifest/ --config-dir examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech_hpu
```
## Supported Configurations

| Device           | SynapseAI Version | PyTorch Version | Mode      |
|------------------|-------------------|-----------------|-----------|
| Gaudi            | 1.9.0             | 1.13.1          | Training  |
| Gaudi2           | 1.9.0             | 1.13.1          | Training  |

## Changelog
### v1.9.0
  - Added HPU graph support to model script. Enabled HPU graph flags for Gaudi2 only.
### Past changes (ported from HabanaAI/model_garden repo)
  - Marked copy to device(inputs) as async.
  - Added async allreduce for sample_size.
  - Removed host barrier in Wav2vec.
  - Replaced isnonzero with where op to unblock the host.
  - Only fetch the log statistics to CPU when needed.
  - Replaced broadcast+sum with equal algorithm to save memory in Quantizer module.
  - Created a customized version of cos_similarity via removing the broadcast operations.
  - Moved negative indices generation to HPU.
  - Changed the data type of randint to int16 to save the memory copyfrom host to device when generating negative indics.
  - Replaced conv1d with equivalent conv2d.
  - Replaced group norm with equivalent instance norm.

### Training Script Modifications
The following are the changes made to the training scripts:

* Added support for Habana devices:
  - Defined certain environment variables Habana devices.
  - Added support to run training in lazy mode in addition to the eager mode.
  - mark_step() is performed to trigger execution.
  - Added support of bucketting, padding, and Precompute loss for HPU.
  - Added support to use HPU accelerator plugin, DDP plugin(for multi-HPU training) and mixed precision plugin.
  - Added support of `fairseq_hydra_train` for multi-node training.
  - Disabled auto dynamic shape support.

## Known Issues
- Only the above configurations mentioned are supported and verified.
- Training on 1 HPU with FP32 data type has OOM issue.