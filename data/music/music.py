import pandas as pd

# lvar, lave, lmax: average, variance, maximum of the frequencies of the left channel
# lfener: an indicator of the amplitude or loudness of the sound
# lfreq: median of the location of the 15 highest peak in the periodogram

def mixed(filepath='data/music/music.csv'):
    df = pd.read_csv(filepath, usecols=[1,2,3,4,5,6,7]) ### Fehler für Spalte 3: singuläre Matrix
    df['lvar'] = df['lvar']/1000
    return df
