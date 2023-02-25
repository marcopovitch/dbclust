# dbclust

Picks clusturing using algorithm such as :

- HDBSCAN (https://github.com/scikit-learn-contrib/hdbscan)
- OPTICS, 
- DBSCAN pick clustering (scikit learn).

The localization is done with NonLinLoc (). The user 
have to provide :

- NLLoc binary file (https://github.com/alomax/NonLinLoc)
- the NonLinLoc configuration template file 
- the time files  

### Requirement
- obspy
- scikit-learn
- hdbscan
- dask
- tqdm
- jinja2
- pandas
