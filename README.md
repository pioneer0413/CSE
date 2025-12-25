# CSE
(placeholder)

## Related Papers
- (placeholder)
- Efficient and Effective Time-Series Forecasting with Spiking Neural Networks
  - ICML '24, (https://arxiv.org/pdf/2402.01533).

## Installation
```
conda create -n SeqSNN python=[3.8, 3.9, 3.10]
conda activate SeqSNN
git clone https://github.com/pioneer0413/CSE
cd CSE
pip install [-e] . 
```
If you would like to make changes and run your experiments, use option `-e`.

## Dataset
(placeholder)

### Data structure
#### Dataset
```
data/
 |- electricity.txt
 |- etth1.txt
 |- etth2.txt
 |- metr-la.txt
 |- solar.txt
 |- weather.txt
```

#### Experimental setup
```
exp/
 |- classification/
 |- detection/
 |- forecast/
     |- dataset/
     |   |- electricity.yml
     |   |- etth1.yml
     |   |- ...
     |- ann/
     |- snn/
```

## Training
(placeholder)

## Acknowledgement
This repo is built upon (Lv's Repo) [https://github.com/microsoft/SeqSNN].
<br>
We show sincerely thanks for @Changze Lv's initial contribution.

## Citation
(placeholder)