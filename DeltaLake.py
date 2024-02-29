import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ## Bronze Schicht

headers = ["risikoniveau","normalisierter-verlustwert","marke","kraftstofftyp","absaugung", "türnummern","körperform",
         "antriebsräder","motorstandort","radstand", "länge","breite","höhe","leergewicht","motortyp",
         "anzahl-der-zylinder", "motorgröße","kraftstoffsystem","bohrung","anschlag","verdichtungsverhältnis","pferdestärken",
         "spitzendrehzahl","stadt-mpg","autobahn-mpg","preis"]

raw_input_df = pd.read_csv('input/auto.csv', names = headers)
display(raw_input_df.head(10))


csv_file_path = './bronze/'
csv_file_name = 'auto_bronze.csv'

if not os.path.exists(csv_file_path):
    os.mkdir(csv_file_path)

raw_input_df.to_csv(csv_file_path + csv_file_name, index=False)


# ## Silber Schicht



bronze_df = pd.read_csv('bronze/auto_bronze.csv')

# Umwandlung von "?" in NaN
bronze_df.replace("?", np.nan, inplace = True)




# Fehlende Werte in jeder Spalte zählen
missing_data = bronze_df.isnull().sum()
missing_data.sort_values(inplace=True, ascending=False)
display(missing_data)




# Ersetzen durch Mittelwert
avg_normalisierter_verlustwert = bronze_df['normalisierter-verlustwert'].astype("float").mean(axis=0)
bronze_df['normalisierter-verlustwert'].replace(np.nan, avg_normalisierter_verlustwert, inplace=True)

avg_bohrung = bronze_df['bohrung'].astype('float').mean(axis=0)
bronze_df['bohrung'].replace(np.nan, avg_bohrung, inplace=True)

avg_anschlag = bronze_df["anschlag"].astype("float").mean(axis = 0)
bronze_df["anschlag"].replace(np.nan, avg_anschlag, inplace = True)

avg_pferdestaerken = bronze_df['pferdestärken'].astype('float').mean(axis=0)
bronze_df['pferdestärken'].replace(np.nan, avg_pferdestaerken, inplace=True)

avg_spitzendrehzahl = bronze_df['spitzendrehzahl'].astype('float').mean(axis=0)
bronze_df['spitzendrehzahl'].replace(np.nan, avg_spitzendrehzahl, inplace=True)


max_tuernummern = bronze_df['türnummern'].value_counts().idxmax()
print(max_tuernummern)
bronze_df['türnummern'].replace(np.nan, max_tuernummern, inplace=True)


bronze_df.dropna(subset=['preis'], axis=0, inplace=True)
bronze_df.reset_index(drop=True, inplace=True)




bronze_df[['bohrung', 'anschlag']] = bronze_df[['bohrung', 'anschlag']].astype("float")
bronze_df['normalisierter-verlustwert'] = bronze_df['normalisierter-verlustwert'].astype("int64")
bronze_df['preis'] = bronze_df['preis'].astype("float")
bronze_df['spitzendrehzahl'] = bronze_df['spitzendrehzahl'].astype("float")
bronze_df['pferdestärken'] = bronze_df['pferdestärken'].astype("int64", copy=True)



bronze_df['stadt-L/100km'] = 235/bronze_df['stadt-mpg']
bronze_df['autobahn-L/100km'] = 235/bronze_df['autobahn-mpg']



bronze_df['länge-norm'] = bronze_df['länge'] / bronze_df['länge'].max()
bronze_df['breite-norm'] = bronze_df['breite'] / bronze_df['breite'].max()
bronze_df['höhe-norm'] = bronze_df['höhe'] / bronze_df['höhe'].max()

display(bronze_df[['länge-norm','breite-norm','höhe-norm']].head())


bins = np.linspace(min(bronze_df['pferdestärken']), max(bronze_df['pferdestärken']), 4)
gruppen_namen = ['niedrig', 'mittel', 'hoch']
bronze_df['pferdestärken-binned'] = pd.cut(bronze_df['pferdestärken'], bins, labels=gruppen_namen, include_lowest=True )
bronze_df[['pferdestärken','pferdestärken-binned']].head(20)



dummy = pd.get_dummies(bronze_df['kraftstofftyp'])
bronze_df = pd.concat([bronze_df, dummy], axis=1)
bronze_df.rename(columns={'gas': 'benzin'}, inplace=True)



parquet_file_path = './silber/'
parquet_file_name = 'auto_silber.parquet'

if not os.path.exists(parquet_file_path):
    os.mkdir(parquet_file_path)

bronze_df.to_parquet(parquet_file_path + parquet_file_name)
display(bronze_df)
print(bronze_df.columns)


# ## Gold Schicht



silber_df = pd.read_parquet('silber/auto_silber.parquet')

def safe_gold_usecase(silber_df, columns, usecase_name):
    parquet_file_path = './gold/'
    parquet_file_name = f'auto_{usecase_name}.parquet'
    
    if not os.path.exists(parquet_file_path):
        os.mkdir(parquet_file_path)
        
    gold_df = silber_df[columns].copy()
    gold_df.to_parquet(parquet_file_path + parquet_file_name)
    display(gold_df)



kraftstoffeffizienz_gold = ['motorgröße', 'kraftstoffsystem', 'bohrung', 'anschlag', 'verdichtungsverhältnis', 'leergewicht', 'pferdestärken', 'spitzendrehzahl', 'stadt-L/100km', 'autobahn-L/100km', 'pferdestärken-binned', 'diesel', 'benzin']

safe_gold_usecase(silber_df, kraftstoffeffizienz_gold, 'kraftstoffeffizienz')




segmentierung_gold  = ['marke', 'normalisierter-verlustwert', 'türnummern', 'körperform', 'antriebsräder', 'motorstandort', 'radstand', 'länge', 'breite', 'höhe', 'leergewicht', 'länge-norm', 'breite-norm', 'höhe-norm']

safe_gold_usecase(silber_df, segmentierung_gold, 'segmentierung')



preisvorhersage_gold  = ['risikoniveau', 'marke', 'kraftstofftyp', 'türnummern', 'antriebsräder', 'motorstandort', 'motortyp', 'anzahl-der-zylinder', 'pferdestärken-binned', 'preis']

safe_gold_usecase(silber_df, preisvorhersage_gold, 'preisvorhersage')
