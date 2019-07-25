# ReCoSa
ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation

Requirement: 
nltk>=3.2.4
numpy>=1.13.0
regex>=2017.6.7
tensorflow>=1.2.0

parameter setting:
hyperparams.py

To train:
python train.py

To eval:
python eval.py

The train data: X X X </d> X X X </d> X X X </d> ...

The answer data: Y Y Y </d>

The vocab data:  word freq

          i.e.   Apple   53
         
                 Banana  23
                 
                 ...
