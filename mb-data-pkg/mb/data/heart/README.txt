## cleaned.cleveland.data
was derived from processed.cleveland.data as follows:

  * add header to file as first line:
age,sex,chest_pain,blood_pres,chol,blood_sugar,restecg,max_heartrate,angina,oldpeak,slope,vessels_cnt,thal,disease
  * for that we use the following name transformations (if any) after the colon  
  
f      -- 1. #3  (age)
c      -- 2. #4  (sex): 
c      -- 3. #9  (cp): chest_pain
f      -- 4. #10 (trestbps): blood_pres
f      -- 5. #12 (chol)
c      -- 6. #16 (fbs): blood_sugar
c      -- 7. #19 (restecg)
f      -- 8. #32 (thalach): max_heartrate
c      -- 9. #38 (exang): angina
f      -- 10. #40 (oldpeak)
c      -- 11. #41 (slope)
i      -- 12. #44 (ca): vessels_cnt
c      -- 13. #51 (thal)
c      -- 14. #58 (num): disease

  * removed all data entries with unknown values, i.e. "?" (by hand, its only a few)
 