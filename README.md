# ReCoSa
ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation\\


Requirement: 

nltk>=3.2.4

numpy>=1.13.0

regex>=2017.6.7

tensorflow>=1.2.0\\


parameter setting:
hyperparams.py\\


To generate vocab:
python prepro.py\\


To train:
python train.py\\


To eval:
python eval.py\\


The dialogue data：Hello How are you?      Good, you?      I'm fine, what's new?

Souce looks like:

Hello How are you?  \</d\>

Hello How are you?  \</d\> Good, you? \</d\>

Hello How are you? \</d\> Good, you? \</d\> I'm fine, what's new?\</d\>


Target:

Good, you?\</d\>

I'm fine, what's new?\</d\>

Nothing much...\</d\>

