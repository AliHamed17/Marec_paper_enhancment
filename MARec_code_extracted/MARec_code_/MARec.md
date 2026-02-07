## MARec

MARec is an approach to leverage semantic features for cold start recommendations. 
We will publicly release the link to the datasets and code so that our results can be reproduced. 
In this folder we share the key components of our code. 


### Files present in this folder

1. MARec.md readme file

2. Mapping between MovieLens/Netflix public movieIDs and rowIDs 
- unique_sid_10m.csv
- unique_sid_netflix.csv


### Notebooks and scripts present in this folder

1. To reproduce results from Tables 4 to 6 (BoW + EASE), pull datasets and run notebooks 
- ML10M-example-notebook.ipynb
- Amazon-example-notebook.ipynb
- MovieLensHetrec-example-notebook.ipynb
- Netflix-example-notebook.ipynb

2. To train a VAE from a pre-trained alignment term, pull datasets and run notebooks:
- ML10M-Aligned-vae.ipynb
- Amazon-Aligned-vae.ipynb
- MovieLensHetrec-Aligned-vae.ipynb

3. To train the Siamese Network
- ML10M-Siamese.ipynb


### External datasets necessary to run the scripts

- the cold start data splits are available at in https://github.com/cesarebernardis/NeuralFeatureCombiner 
- the ML10M metadata is available at https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset 
- the Netflix metadata is available by joining Netflix titles with Ibmd titles, with the Imdb metadata at https://datasets.imdbws.com/ 
