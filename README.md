# text-classification-meituan
Course project for CS420


## Usage

MLP, tfidf feature
```
python main.py --model MLP --d_feature 10000 --d_hidden 500 --n_layers 2 --device 1 --n_epochs 2
```

BiLSTM, word vector
```
python main.py --model BiLSTM --d_feature 300 
```