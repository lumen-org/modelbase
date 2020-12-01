## origin of data set
from the ggobi data sets

## olive.original.data
the original data

## olive.csv
derived from olive.original.data by the following procedure:

 1. replaces first column (index of observation) using a regular expression replacement:
   replace: "[\d]*\."
   by: ""
 2. remove the leading "," in the header (the header entry for the index)
 3. replace the header by:
"area,region-int,area-int,palmitic,palmitoleic,stearic,oleic,linoleic,linolenic,arachidic,eicosenoic"
 
## notes:
 * region is a superclass of area: "Three “super-classes” of Italy: North (3), South(1), and the island
of Sardinia(2)"
 * area is a integer encoding for the region of origin: "Nine collection areas: three from the region North (Umbria,East and West Liguria), four from South (North and South Apulia, Calabria, and Sicily), and two from the island of Sardinia (inland and coastal Sardinia).
