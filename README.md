
# AIN: Fast and Accurate Sequence Labeling with Approximate Inference Network

The code is mainly for our EMNLP 2020 paper: [AIN: Fast and Accurate Sequence Labeling with Approximate Inference Network](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp20ain.pdf).
The code is mainly based on [MultilangStructureKD](https://github.com/Alibaba-NLP/MultilangStructureKD) with a lot of modifications.

AINs approximate the CRF inference steps. Instead of Forward-backward algorithm and Viterbi algorithm that requires sequential compuation over the input sequence, we use Mean-Field Variational Inference algorithm to approximate the CRF inference which can be paralleled over the sequence. In the paper, we show that AIN on the first-order CRF is about 10.2 and 4.4 times faster than the traditional CRF approach in training and prediction respectively with long sentences. Moreover, AINs achieves competitive accuracy with the tranditional CRF approach. Please refer to the paper for more details.

---

## Requirements

The project is based on PyTorch 1.1+ and Python 3.6+. We create the virtual environment based on [anaconda](https://www.anaconda.com/) (The requirements are directly extracted from my environment, therefore there exists some non-essential packages like TensorFlow, we will update the requirements in the future):

```
conda create --name parser --file requirements2.txt
source activate parser
pip install -r requirements.txt
```

## Training
To train AIN on the first-order CRF, run:
```
python -u train.py --config config/3iter_word_char_charcnn_300epoch_32batch_0.1lr_1window_256hidden_en_lample_monolingual_mfvi_sentloss_10patience_baseline_fast_2nd_startend_sentbatch_norelearn_nodev_ner35.yaml
```

To train AIN on the factorized second-order CRF, run:
```
python -u train.py --config config/3iter_word_char_charcnn_300epoch_32batch_0.1lr_2window_256hidden_en_lample_monolingual_mfvi_sentloss_10patience_baseline_fast_2nd_startend_sentbatch_norelearn_nodev_ner35.yaml
```

## Reference

[AIN: Fast and Accurate Sequence Labeling with Approximate Inference Network](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp20ain.pdf)


## Contact 

Please email your questions or comments to [Xinyu Wang](http://wangxinyu0922.github.io/).

