require(plyr)

mpg1 <- read.csv(file='mpg98-08.csv') # 98-08
mpg2 <- read.csv(file='mpg85-97.csv') # 85-97
vars <- c('year','class','cyl','displ','cty','hwy')
mpg2$displ <- mpg2$displ*.0164 # adjusting for cubic inches
mpg_full <- rbind(mpg2,mpg1)

mpg_full$class <- revalue( mpg_full$class, c(
  "compact cars"="compact car",
  "minicompact cars"="compact car",
  "subcompact cars"="compact car",
  "midsize cars"="midsize car",
  "minivan"="large car",
  "large cars"="large car",
  "two seaters"="two seater",
  "small pickup trucks 2wd"="pickup",
  "standard pickup trucks 2wd"="pickup",
  "small pickup trucks 4wd"="pickup",
  "standard pickup trucks 4wd"="pickup",
  "small station wagons"="station wagon",
  "midsize station wagons"="station wagon",
  "large station wagon"="station wagon",
  "special purpose vechicles cab chassis"="spv",
  "special purpose vehicles 2wd"="spv",
  "special purpose vehicles 4wd"="spv",
  "vans, cargo type"="cargo van",
  "vans, passenger type"="passenger van",
  "s.u.v. - 2wd"="suv",
  "s.u.v. - 4wd"="suv"))

write.csv(mpg_full, file = 'mpg.csv')

