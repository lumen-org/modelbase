## Origin of data set:
see other readme file
 
## concrete.data.ssv
original data

## concrete.csv
derived from concrete.data.ssv as follows:


 1. removed empty trailing line manually
 2. made it a csv file by regex replacement:
  find: "[ ]*" 
  replace by: ","
 3. removed trailing "," at the end of each line:
   find: ",\n"
   replace: "\n"
 4. added header manually:
"Cement,Blast,Fly_Ash,Water,Superplasticizer,Coarse_Aggregate,Fine_Aggregate,Age,compr_strgth"

## notes on variables:

see Concrete_Readme.txt

