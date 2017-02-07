import pandas as pd

#tars1: width of the first joint of the first tarsus in microns
#tars2: _____________second_____________first__________________
#head: the maximal width of the head between the external edges of the exes in 0.01mm
#aede1: the maximal wodth of the aedeagus in the fore-part in microns
#aede2: the front angle of the aedeagus (1unit = 7.5 degree)
#aede3: the aedeagus width fromthe side in microns

def mixed(filepath='data/flea/flea.csv'):
    df = pd.read_csv(filepath) 
    return df
	
if __name__ == '__main__':
	df = mixed('flea.csv')
	print(df.head())
