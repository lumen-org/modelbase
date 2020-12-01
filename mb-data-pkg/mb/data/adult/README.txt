## adult.full
adult.full was derived from adult.data and adult.test as follows:

  1. both files where concatenated (any empty lines removed)
  2. any occurance of ">50K." was replaced by ">50K"
  3. any occurance of "<=50K." was replaced by "<=50K"
  4. the following header line was inserted as the first line of the file (with the quotes)
    "age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country"

Notes:  
 * 2 and 3 are because adult.test differs from adult.data in this regard
 * the header line is directly derived from the adult.names file
	
## adult.full.cleansed
this file was created from adult.full as follows
	
  1. all lines with unknown values are removed, by running this regex command: ^.*\?.*$ (replacing it with the empty string)
  2. all possibly existing existing empty line are removed by this regex: ^[\n\r]+ (replacing it with the empty string)
