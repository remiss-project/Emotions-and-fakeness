# Emotions-and-fakeness

# Model de probabilitat de faula

## RFmodel.joblib
Amb este model es vol obtindre la ***probabilitat*** que un ***tuit polític*** siga ***faula*** segons la seua redacció, és a dir, tenint en compte el contingut sintàctica i les emocions contenen.

## emotions_rf_extraction.py
```
python3 emotions_rf_extraction.py -i INPUT-CSV-PATH -o OUTPUT-CSV-PATH -rf RF-PATH -c TEXT-COL
```

Aquest codi rep d'entrada:
- (`INPUT-CSV-PATH`) la ruta on es troba el csv a processar
- (`OUTPUT-CSV-PATH`) la ruta del nou csv amb els resultats
- (`RF-PATH`) la ruta on es troba el model Random-Forest
- (`TEXT-COL`) el nom de la columna del csv amb els textos a analitzar.

El codi, aplica diferents models pre-entrenats per a obtindre les emocions i sentiments, i llibreries per a analitzar la informació sintàctica i poder utilitzar-les per a calcular (amb el model Random-Forest) la probabilitat que els tuits siguen una faula.

Exemple d'us, on ***text*** és el nombre de la columna amb els textos a analitzar:
```
python3 emotions_rf_extraction.py -i /home/tweets_eleccions.csv -o /home/tweets_eleccions_rf.csv -rf /home/RFmodel.joblib -c text
```
