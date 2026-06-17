# DSF–MarianMT: Dynamic Semantic Fusion Enhanced Neural Machine Translation

## Overview
Implementation of DSF–MarianMT for Chinese–English Neural Machine Translation.

Models:
- Word2Vec + Bi-LSTM
- Word2Vec + GRU
- Word2Vec + CNN
- Sentence Embedding + Bi-LSTM
- Sentence Embedding + GRU
- Sentence Embedding + CNN
- BERT Encoder-Decoder
- mBERT Encoder-Decoder
- mBERT + Decoder
- MarianMT
- DSF–MarianMT

## Dataset
CSV format:

chinese,english
我喜欢机器翻译。,I like machine translation.

Dataset split:
- Training: 80%
- Validation: 10%
- Testing: 10%

Random seed:
42

## Installation

pip install torch
pip install transformers
pip install sentence-transformers
pip install sacrebleu
pip install gensim
pip install scipy
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib

Optional:
pip install unbabel-comet

## Training

DSF–MarianMT:

python train_dsf_marianmt_full_real.py --data translation.csv --run_dsf --epochs 70

Transformer baselines:

python train_dsf_marianmt_full_real.py --data translation.csv --run_transformers --epochs 70

Classical baselines:

python train_dsf_marianmt_full_real.py --data translation.csv --run_classic --epochs 70

Complete pipeline:

python train_dsf_marianmt_full_real.py --data translation.csv --run_all --epochs 70

## Dynamic Semantic Fusion

Fusion combines:
- Encoder hidden representation
- Word-level semantic representation
- Sentence-level semantic representation

through a learnable gating mechanism.

## Contrastive Semantic Learning

Positive pairs:
Aligned source-target sentence pairs.

Negative pairs:
Random in-batch source-target mismatches.

## Evaluation

BLEU:
- SacreBLEU
- Case-sensitive
- tokenizer=13a

Additional metrics:
- chrF
- TER
- COMET

## Statistical Validation

Automatically computes:
- Bootstrap Resampling
- Paired t-test
- Wilcoxon Signed-Rank Test
- Cohen's d

Output:
statistical_validation.csv

## Ablation Studies

Generated configurations:
- DSF_no_word
- DSF_no_sentence
- DSF_no_contrastive
- DSF–MarianMT

Output:
fusion_component_ablation.csv

## Explainability

Generated:
- Attention Entropy Analysis
- Attention Head Importance
- Component Contribution Analysis

Output:
Fig16_XAI_Component_Importance.png

## Output Structure

output_dir/
├── train_split_seed42.csv
├── validation_split_seed42.csv
├── test_split_seed42.csv
├── figures/
├── tables/
└── models/

## Reproducibility

All experiments use:
Random Seed = 42
