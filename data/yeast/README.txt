## Origin of data set:
https://archive.ics.uci.edu/ml/datasets/Yeast
 
## yeast.csv
Derived from yeast.data by the following procedure:

 1. replacing sequences of whitespaces (" ") by a single comma using regex: "[]*"  replaced by  ","
 2. manually added the header:
name,mcg,gvh,alm,mit,erl,pox,vac,nuc,location

## notes:

categorical dimensions: name,erl,pox,location
continuous dimensions: mcg,gvh,alm,mit,vac,nuc

erl takes only the values 1.00 or 0.50
pox takes only the values 0.00 or 0.50

