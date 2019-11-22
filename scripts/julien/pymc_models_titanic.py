#!usr/bin/python
# -*- coding: utf-8 -*-import string

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import theano

import pandas as pd

import math
import os

from mb_modelbase.models_core.pyMC3_model import ProbabilisticPymc3Model
from mb_modelbase.utils.data_type_mapper import DataTypeMapper

filepath = os.path.join(os.path.dirname(__file__), "titanic_original.csv")
df = pd.read_csv(filepath)
sample_size = 1000

titanic_backward_map = {
    'pclass': {0: 1, 1: 2, 2: 3},
    'survived': {0: 0, 1: 1},
    'sex': {0: 'female', 1: 'male'},
    'age': {0: 0.1667, 1: 0.4167, 2: 0.6667, 3: 0.75, 4: 0.8333, 5: 0.9167, 6: 1.0, 7: 2.0, 8: 3.0, 9: 4.0, 10: 5.0, 11: 6.0, 12: 7.0, 13: 8.0, 14: 9.0, 15: 11.0, 16: 12.0, 17: 13.0, 18: 14.0, 19: 15.0, 20: 16.0, 21: 17.0, 22: 18.0, 23: 19.0, 24: 20.0, 25: 21.0, 26: 22.0, 27: 23.0, 28: 24.0, 29: 25.0, 30: 26.0, 31: 27.0, 32: 28.0, 33: 29.0, 34: 29.881134512428304, 35: 30.0, 36: 31.0, 37: 32.0, 38: 32.5, 39: 33.0, 40: 34.0, 41: 35.0, 42: 36.0, 43: 36.5, 44: 37.0, 45: 38.0, 46: 39.0, 47: 40.0, 48: 41.0, 49: 42.0, 50: 43.0, 51: 44.0, 52: 45.0, 53: 47.0, 54: 48.0, 55: 49.0, 56: 50.0, 57: 51.0, 58: 52.0, 59: 53.0, 60: 54.0, 61: 55.0, 62: 56.0, 63: 58.0, 64: 59.0, 65: 60.0, 66: 62.0, 67: 63.0, 68: 64.0, 69: 76.0, 70: 80.0},
    'sibsp': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    'parch': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    'ticket': {0: '110152', 1: '110413', 2: '110564', 3: '110813', 4: '111163', 5: '111361', 6: '111369', 7: '111426', 8: '111427', 9: '111428', 10: '112053', 11: '112058', 12: '112277', 13: '112377', 14: '112378', 15: '112901', 16: '113055', 17: '113503', 18: '113505', 19: '113509', 20: '113572', 21: '113760', 22: '113773', 23: '113776', 24: '113780', 25: '113781', 26: '113783', 27: '113786', 28: '113788', 29: '113789', 30: '113794', 31: '113795', 32: '113798', 33: '113803', 34: '113804', 35: '113806', 36: '11751', 37: '11752', 38: '11753', 39: '11755', 40: '11765', 41: '11767', 42: '11769', 43: '11770', 44: '11774', 45: '11778', 46: '11813', 47: '11967', 48: '12749', 49: '13050', 50: '13213', 51: '13214', 52: '13236', 53: '13502', 54: '13507', 55: '13508', 56: '13567', 57: '13568', 58: '13695', 59: '14311', 60: '14312', 61: '14313', 62: '1601', 63: '16966', 64: '16988', 65: '17421', 66: '17453', 67: '17464', 68: '17465', 69: '17466', 70: '17474', 71: '17765', 72: '17770', 73: '19877', 74: '19928', 75: '19943', 76: '19947', 77: '19950', 78: '19952', 79: '19988', 80: '19996', 81: '2003', 82: '21228', 83: '220844', 84: '220845', 85: '223596', 86: '226593', 87: '226875', 88: '228414', 89: '230080', 90: '230136', 91: '230433', 92: '230434', 93: '234604', 94: '234818', 95: '236853', 96: '237668', 97: '237789', 98: '237798', 99: '24160', 100: '243847', 101: '243880', 102: '244270', 103: '244367', 104: '244373', 105: '248698', 106: '248706', 107: '248727', 108: '248733', 109: '248738', 110: '250644', 111: '250648', 112: '250649', 113: '250652', 114: '250655', 115: '2543', 116: '2620', 117: '2625', 118: '26360', 119: '2649', 120: '2650', 121: '2651', 122: '2653', 123: '2654', 124: '2657', 125: '2658', 126: '2659', 127: '2661', 128: '2663', 129: '2666', 130: '2667', 131: '2668', 132: '26707', 133: '2677', 134: '2687', 135: '2688', 136: '2696', 137: '2699', 138: '27042', 139: '27267', 140: '27849', 141: '28034', 142: '28220', 143: '28404', 144: '28551', 145: '2908', 146: '29103', 147: '29105', 148: '29106', 149: '2926', 150: '29750', 151: '3101265', 152: '3101278', 153: '3101281', 154: '3101298', 155: '31027', 156: '312991', 157: '31418', 158: '315098', 159: '315153', 160: '323592', 161: '330919', 162: '330920', 163: '330931', 164: '330932', 165: '330958', 166: '330980', 167: '334914', 168: '334915', 169: '335432', 170: '335677', 171: '33638', 172: '34218', 173: '342712', 174: '343120', 175: '345501', 176: '345572', 177: '345768', 178: '345774', 179: '345779', 180: '3470', 181: '347066', 182: '347070', 183: '347072', 184: '347077', 185: '347079', 186: '347081', 187: '347083', 188: '347085', 189: '347089', 190: '347469', 191: '347470', 192: '347742', 193: '348122', 194: '348125', 195: '349240', 196: '349256', 197: '349910', 198: '350026', 199: '350033', 200: '350034', 201: '350043', 202: '350046', 203: '350053', 204: '350054', 205: '350403', 206: '350417', 207: '35273', 208: '35851', 209: '35852', 210: '363291', 211: '364516', 212: '36568', 213: '367226', 214: '367230', 215: '367231', 216: '368402', 217: '36866', 218: '36928', 219: '36947', 220: '36973', 221: '370370', 222: '370373', 223: '370375', 224: '371110', 225: '371362', 226: '374887', 227: '376564', 228: '382650', 229: '382651', 230: '382653', 231: '383123', 232: '383162', 233: '386525', 234: '392091', 235: '392096', 236: '4134', 237: '4138', 238: '65306', 239: '7538', 240: '7548', 241: '7598', 242: '9234', 243: 'A. 2. 39186', 244: 'A/4 31416', 245: 'A/5 3540', 246: 'A/5. 10482', 247: 'C 17368', 248: 'C 17369', 249: 'C 7077', 250: 'C.A. 2315', 251: 'C.A. 2673', 252: 'C.A. 29395', 253: 'C.A. 31026', 254: 'C.A. 31921', 255: 'C.A. 33112', 256: 'C.A. 33595', 257: 'C.A. 34260', 258: 'C.A. 34644', 259: 'C.A. 34651', 260: 'C.A. 37671', 261: 'CA 31352', 262: 'CA. 2314', 263: 'F.C. 12750', 264: 'F.C. 12998', 265: 'F.C.C. 13528', 266: 'F.C.C. 13529', 267: 'F.C.C. 13531', 268: 'LINE', 269: 'P/PP 3381', 270: 'PC 17473', 271: 'PC 17474', 272: 'PC 17475', 273: 'PC 17476', 274: 'PC 17477', 275: 'PC 17482', 276: 'PC 17483', 277: 'PC 17485', 278: 'PC 17558', 279: 'PC 17569', 280: 'PC 17572', 281: 'PC 17582', 282: 'PC 17585', 283: 'PC 17592', 284: 'PC 17594', 285: 'PC 17597', 286: 'PC 17598', 287: 'PC 17599', 288: 'PC 17600', 289: 'PC 17603', 290: 'PC 17604', 291: 'PC 17606', 292: 'PC 17607', 293: 'PC 17608', 294: 'PC 17610', 295: 'PC 17611', 296: 'PC 17613', 297: 'PC 17755', 298: 'PC 17756', 299: 'PC 17757', 300: 'PC 17758', 301: 'PC 17759', 302: 'PC 17760', 303: 'PC 17761', 304: 'PP 9549', 305: 'S.C./PARIS 2079', 306: 'S.O./P.P. 2', 307: 'S.O./P.P. 752', 308: 'S.W./PP 752', 309: 'SC 1748', 310: 'SC/AH Basle 541', 311: 'SC/PARIS 2146', 312: 'SC/PARIS 2147', 313: 'SC/PARIS 2148', 314: 'SC/PARIS 2149', 315: 'SC/PARIS 2166', 316: 'SC/PARIS 2167', 317: 'SC/Paris 2123', 318: 'SOTON/O.Q. 3101308', 319: 'SOTON/O.Q. 392078', 320: 'SOTON/O2 3101284', 321: 'SOTON/OQ 392089', 322: 'STON/O 2. 3101269', 323: 'STON/O 2. 3101285', 324: 'STON/O 2. 3101286', 325: 'STON/O 2. 3101288', 326: 'STON/O 2. 3101289', 327: 'STON/O2. 3101279', 328: 'SW/PP 751', 329: 'W./C. 14258', 330: 'W./C. 14260', 331: 'W./C. 14266', 332: 'W.E.P. 5734', 333: 'WE/P 5735'},
    'fare': {0: 0.0, 1: 3.1708, 2: 6.95, 3: 6.975, 4: 7.05, 5: 7.1417, 6: 7.225, 7: 7.2292, 8: 7.25, 9: 7.4958, 10: 7.55, 11: 7.5792, 12: 7.65, 13: 7.7208, 14: 7.725, 15: 7.7333, 16: 7.7375, 17: 7.75, 18: 7.775, 19: 7.7875, 20: 7.7958, 21: 7.8208, 22: 7.8292, 23: 7.8542, 24: 7.8792, 25: 7.8875, 26: 7.8958, 27: 7.925, 28: 8.05, 29: 8.1125, 30: 8.5167, 31: 8.6625, 32: 8.9625, 33: 9.225, 34: 9.35, 35: 9.5, 36: 9.5875, 37: 9.8417, 38: 10.5, 39: 11.1333, 40: 11.2417, 41: 12.0, 42: 12.2875, 43: 12.35, 44: 12.475, 45: 12.7375, 46: 13.0, 47: 13.4167, 48: 13.5, 49: 13.7917, 50: 13.8583, 51: 13.8625, 52: 13.9, 53: 14.1083, 54: 14.4542, 55: 14.5, 56: 15.2458, 57: 15.5, 58: 15.55, 59: 15.7417, 60: 15.75, 61: 15.85, 62: 15.9, 63: 16.0, 64: 16.1, 65: 16.7, 66: 17.4, 67: 18.75, 68: 18.7875, 69: 19.2583, 70: 19.5, 71: 20.25, 72: 20.525, 73: 20.575, 74: 21.0, 75: 22.025, 76: 22.3583, 77: 23.0, 78: 23.25, 79: 24.0, 80: 24.15, 81: 25.7, 82: 25.7417, 83: 25.9292, 84: 26.0, 85: 26.25, 86: 26.2833, 87: 26.2875, 88: 26.3875, 89: 26.55, 90: 27.0, 91: 27.4458, 92: 27.7208, 93: 27.75, 94: 28.5, 95: 28.5375, 96: 29.0, 97: 29.7, 98: 30.0, 99: 30.5, 100: 30.6958, 101: 31.0, 102: 31.3875, 103: 31.6833, 104: 32.5, 105: 33.0, 106: 35.5, 107: 36.75, 108: 37.0042, 109: 39.0, 110: 39.4, 111: 39.6, 112: 41.5792, 113: 49.5, 114: 49.5042, 115: 51.4792, 116: 51.8625, 117: 52.0, 118: 52.5542, 119: 53.1, 120: 55.0, 121: 55.4417, 122: 55.9, 123: 56.4958, 124: 56.9292, 125: 57.0, 126: 57.75, 127: 57.9792, 128: 59.4, 129: 60.0, 130: 61.175, 131: 61.3792, 132: 61.9792, 133: 63.3583, 134: 65.0, 135: 66.6, 136: 69.3, 137: 71.0, 138: 71.2833, 139: 75.2417, 140: 75.25, 141: 76.2917, 142: 76.7292, 143: 77.9583, 144: 78.2667, 145: 78.85, 146: 79.2, 147: 79.65, 148: 80.0, 149: 81.8583, 150: 82.1708, 151: 82.2667, 152: 83.1583, 153: 83.475, 154: 86.5, 155: 89.1042, 156: 90.0, 157: 91.0792, 158: 93.5, 159: 106.425, 160: 108.9, 161: 110.8833, 162: 113.275, 163: 120.0, 164: 133.65, 165: 134.5, 166: 135.6333, 167: 136.7792, 168: 146.5208, 169: 151.55, 170: 153.4625, 171: 164.8667, 172: 211.3375, 173: 211.5, 174: 221.7792, 175: 227.525, 176: 247.5208, 177: 262.375, 178: 263.0, 179: 512.3292},
    'embarked': {0: 'C', 1: 'Q', 2: 'S'},
    'boat': {0: '1', 1: '10', 2: '11', 3: '12', 4: '13', 5: '13 15', 6: '13 15 B', 7: '14', 8: '15', 9: '15 16', 10: '16', 11: '2', 12: '3', 13: '4', 14: '5', 15: '5 7', 16: '5 9', 17: '6', 18: '7', 19: '8', 20: '8 10', 21: '9', 22: 'A', 23: 'B', 24: 'C', 25: 'C D', 26: 'D'},
    'has_cabin_number': {0: 0, 1: 1},
}

dtm = DataTypeMapper()
for name, map_ in titanic_backward_map.items():
    dtm.set_map(forward='auto', backward=map_, name=name)

###############################################################
# 158 parameter
# continuous_variables = ['age', 'fare', 'ticket']
###############################################################
def create_titanic_model_1(filename="", modelname="titanic_model_1", fit=True):
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables
    import pymc3 as pm
    titanic_model = pm.Model()
    data = None
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

#####################
# continuous_variables = ["ticket"]
# 376 parameter
#####################
def create_titanic_model_2(filename="", modelname="titanic_model_2", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        age = pm.Categorical('age',
                             p=[0.0021, 0.0021, 0.0021, 0.0041, 0.0062, 0.0041, 0.0144, 0.0082, 0.0103, 0.0144, 0.0082,
                                0.0062, 0.0041, 0.0082, 0.0082, 0.0021, 0.0062, 0.0062, 0.0062, 0.0062, 0.0165, 0.0123,
                                0.0267, 0.0226, 0.0165, 0.0226, 0.0412, 0.0185, 0.0453, 0.0247, 0.0206, 0.0267, 0.0144,
                                0.0267, 0.142, 0.0309, 0.0226, 0.0247, 0.0021, 0.0165, 0.0123, 0.0267, 0.0329, 0.0021,
                                0.0041, 0.0123, 0.0165, 0.0123, 0.0041, 0.0082, 0.0062, 0.0062, 0.0288, 0.0041, 0.0206,
                                0.0103, 0.0123, 0.0062, 0.0062, 0.0082, 0.0103, 0.0082, 0.0041, 0.0062, 0.0021, 0.0082,
                                0.0041, 0.0041, 0.0041, 0.0021, 0.0021])
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Categorical('fare',
                              p=[0.0041, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0103, 0.0123, 0.0062, 0.0021, 0.0041,
                                 0.0021, 0.0062, 0.0021, 0.0021, 0.0062, 0.0021, 0.0309, 0.0144, 0.0021, 0.0103, 0.0021,
                                 0.0021, 0.0082, 0.0062, 0.0021, 0.0021, 0.0144, 0.0165, 0.0021, 0.0041, 0.0021, 0.0021,
                                 0.0021, 0.0041, 0.0062, 0.0021, 0.0021, 0.0226, 0.0062, 0.0041, 0.0021, 0.0041, 0.0041,
                                 0.0082, 0.0021, 0.0329, 0.0041, 0.0041, 0.0021, 0.0062, 0.0041, 0.0021, 0.0021, 0.0021,
                                 0.0041, 0.0123, 0.0062, 0.0041, 0.0062, 0.0041, 0.0041, 0.0062, 0.0021, 0.0062, 0.0062,
                                 0.0041, 0.0062, 0.0021, 0.0082, 0.0041, 0.0021, 0.0041, 0.0062, 0.0103, 0.0062, 0.0062,
                                 0.0103, 0.0062, 0.0021, 0.0021, 0.0021, 0.0021, 0.0041, 0.037, 0.0082, 0.0021, 0.0062,
                                 0.0021, 0.0206, 0.0021, 0.0021, 0.0123, 0.0062, 0.0021, 0.0021, 0.0062, 0.0041, 0.0123,
                                 0.0082, 0.0021, 0.0041, 0.0062, 0.0021, 0.0041, 0.0041, 0.0062, 0.0062, 0.0041, 0.0123,
                                 0.0041, 0.0041, 0.0062, 0.0021, 0.0021, 0.0041, 0.0021, 0.0082, 0.0082, 0.0082, 0.0041,
                                 0.0082, 0.0021, 0.0123, 0.0041, 0.0041, 0.0041, 0.0041, 0.0062, 0.0021, 0.0021, 0.0021,
                                 0.0021, 0.0041, 0.0062, 0.0021, 0.0041, 0.0021, 0.0021, 0.0021, 0.0021, 0.0041, 0.0062,
                                 0.0062, 0.0041, 0.0041, 0.0062, 0.0041, 0.0041, 0.0062, 0.0021, 0.0041, 0.0103, 0.0021,
                                 0.0062, 0.0041, 0.0082, 0.0041, 0.0041, 0.0041, 0.0041, 0.0062, 0.0041, 0.0082, 0.0041,
                                 0.0103, 0.0062, 0.0021, 0.0021, 0.0062, 0.0041, 0.0062, 0.0082, 0.0041, 0.0021, 0.0062,
                                 0.0041, 0.0123, 0.0082, 0.0082])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        ticket = pm.Normal('ticket',
                           mu=tt.switch(tt.eq(pclass, 0), 128.8706, tt.switch(tt.eq(pclass, 1), 175.9464, 191.1503)),
                           sigma=tt.switch(tt.eq(pclass, 0), 115.7033, tt.switch(tt.eq(pclass, 1), 86.3896, 61.4168)))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))

        #data = pm.trace_to_dataframe(pm.sample(10000))
        #data.sort_index(inplace=True)

    m = ProbabilisticPymc3Model(modelname, titanic_model, data_mapping=dtm)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m

###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# 158 parameter
###############################################################
def create_titanic_model_3(filename="", modelname="titanic_model_3", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    # define model explicitely (no fitting to data with pymc3)
    titanic_model = pm.Model()
    with titanic_model:
        survived = pm.Categorical('survived', p=[0.0185, 0.9815])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(survived, 0), [0.1111, 0.8889], [0.6667, 0.3333]))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        pclass = pm.Categorical('pclass',
                                p=tt.switch(tt.eq(sex, 0), [0.4326, 0.2696, 0.2978], [0.3772, 0.1557, 0.4671]))
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    # import pandas as pd

    # filepath = os.path.join(os.path.dirname(__file__), "titanic_cleaned.csv")
    # df = pd.read_csv(filepath)
    if fit:
        m.fit(df, auto_extend=False)
    return df, m



###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# whitelist = [("pclass", "survived"), ("sex", "survived")]
# 159 parameter
###############################################################
def create_titanic_model_4(filename="", modelname="titanic_model_4", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    data = None
    with titanic_model:
        pclass = pm.Categorical('pclass', p=[0.4136, 0.2305, 0.356])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(pclass, 0), [0.6866, 0.3134],
                                                tt.switch(tt.eq(pclass, 1), [0.7679, 0.2321], [0.5491, 0.4509])))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        survived = pm.Categorical('survived', p=tt.switch(tt.eq(sex, 0), [0.0031, 0.9969], [0.0479, 0.9521]))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))

    m = ProbabilisticPymc3Model(modelname, titanic_model)
    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m


###############################################################
# continuous_variables = ['age', 'fare', 'ticket']
# whitelist = [("pclass", "survived"), ("sex", "survived")]
# 167 parameter
###############################################################
def create_titanic_model_5(filename="", modelname="titanic_model_5", fit=True):
    # income is gaussian, depends on age
    if fit:
        modelname = modelname + '_fitted'
    # Set up shared variables

    titanic_model = pm.Model()
    data = None
    with titanic_model:
        pclass = pm.Categorical('pclass', p=[0.4136, 0.2305, 0.356])
        sex = pm.Categorical('sex', p=tt.switch(tt.eq(pclass, 0), [0.6866, 0.3134],
                                                tt.switch(tt.eq(pclass, 1), [0.7679, 0.2321], [0.5491, 0.4509])))
        sibsp = pm.Categorical('sibsp', p=[0.6111, 0.3354, 0.0391, 0.0082, 0.0062])
        parch = pm.Categorical('parch', p=[0.6667, 0.2016, 0.1173, 0.0103, 0.0021, 0.0021])
        fare = pm.Normal('fare', mu=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 108.8666,
                                                                       tt.switch(tt.eq(pclass, 1), 24.1183, 12.5147)),
                                              tt.switch(tt.eq(pclass, 0), 72.0712,
                                                        tt.switch(tt.eq(pclass, 1), 20.2144, 14.015))),
                         sigma=tt.switch(tt.eq(sex, 0), tt.switch(tt.eq(pclass, 0), 83.3719,
                                                                  tt.switch(tt.eq(pclass, 1), 11.9473, 5.838)),
                                         tt.switch(tt.eq(pclass, 0), 91.1007,
                                                   tt.switch(tt.eq(pclass, 1), 9.2129, 13.3239))))
        embarked = pm.Categorical('embarked', p=tt.switch(tt.eq(pclass, 0), [0.4876, 0.01, 0.5025],
                                                          tt.switch(tt.eq(pclass, 1), [0.1339, 0.0179, 0.8482],
                                                                    [0.2081, 0.1965, 0.5954])))
        boat = pm.Categorical('boat', p=tt.switch(tt.eq(pclass, 0),
                                                  [0.0249, 0.0398, 0.0299, 0.0, 0.005, 0.0, 0.0, 0.0249, 0.005, 0.0,
                                                   0.0, 0.0348, 0.1294, 0.1194, 0.1343, 0.01, 0.005, 0.0945, 0.1095,
                                                   0.1144, 0.005, 0.0299, 0.0149, 0.0149, 0.01, 0.0, 0.0448],
                                                  tt.switch(tt.eq(pclass, 1),
                                                            [0.0, 0.1339, 0.125, 0.1518, 0.1071, 0.0, 0.0, 0.2054,
                                                             0.0089, 0.0, 0.0268, 0.0, 0.0, 0.0625, 0.0, 0.0, 0.0, 0.0,
                                                             0.0089, 0.0, 0.0, 0.1429, 0.0, 0.0089, 0.0, 0.0, 0.0179],
                                                            [0.0, 0.0347, 0.0289, 0.0116, 0.1503, 0.0116, 0.0058,
                                                             0.0289, 0.2023, 0.0058, 0.1156, 0.0347, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0058, 0.0, 0.0, 0.0, 0.0173, 0.0462, 0.0289, 0.2081,
                                                             0.0116, 0.052])))
        has_cabin_number = pm.Categorical('has_cabin_number', p=tt.switch(tt.eq(pclass, 0), [0.1692, 0.8308],
                                                                          tt.switch(tt.eq(pclass, 1), [0.8482, 0.1518],
                                                                                    [0.948, 0.052])))
        survived = pm.Categorical('survived',
                                  p=tt.switch(tt.eq(pclass, 0), tt.switch(tt.eq(sex, 0), [0.0, 1.0], [0.0317, 0.9683]),
                                              tt.switch(tt.eq(pclass, 1),
                                                        tt.switch(tt.eq(sex, 0), [0.0, 1.0], [0.0385, 0.9615]),
                                                        tt.switch(tt.eq(sex, 0), [0.0105, 0.9895], [0.0641, 0.9359]))))
        age = pm.Normal('age', mu=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 36.3462,
                                                                          tt.switch(tt.eq(pclass, 1), 20.5921,
                                                                                    20.5546)),
                                            tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 35.0,
                                                                                    tt.switch(tt.eq(pclass, 1), 29.9406,
                                                                                              27.8244)),
                                                      tt.switch(tt.eq(pclass, 0), 35.7486,
                                                                tt.switch(tt.eq(pclass, 1), 25.458, 23.4376)))),
                        sigma=tt.switch(tt.eq(embarked, 0), tt.switch(tt.eq(pclass, 0), 12.6923,
                                                                      tt.switch(tt.eq(pclass, 1), 10.5268, 11.7445)),
                                        tt.switch(tt.eq(embarked, 1), tt.switch(tt.eq(pclass, 0), 2.8284,
                                                                                tt.switch(tt.eq(pclass, 1), 0.0841,
                                                                                          4.2584)),
                                                  tt.switch(tt.eq(pclass, 0), 14.5895,
                                                            tt.switch(tt.eq(pclass, 1), 14.4234, 11.2448)))))
        ticket = pm.Normal('ticket', mu=tt.switch(tt.eq(pclass, 0), fare * 0.4452 + 85.533,
                                                  tt.switch(tt.eq(pclass, 1), fare * -1.1782 + 203.294,
                                                            fare * -2.3243 + 221.8102)),
                           sigma=tt.switch(tt.eq(pclass, 0), 109.2471, tt.switch(tt.eq(pclass, 1), 85.7164, 57.0786)))


    data_map = {}

    raise NotImplementedError("Data Map")
    m = ProbabilisticPymc3Model(modelname, titanic_model, data_map)

    """
    nodes = ['age', 'sex', ...]
    edges = [('age', 'sex'), ...]
    blacklist
    whitelist
    continuous
    """

    m.set_gm_graph()


    m.nr_of_posterior_samples = sample_size
    if fit:
        m.fit(df, auto_extend=False)
    return df, m


if __name__ == '__main__':
    # create original data from backmapping
    from mb_modelbase.utils.data_type_mapper import DataTypeMapper
    dtm = DataTypeMapper()
    for name, map_ in titanic_backward_map.items():
        dtm.set_map(forward='auto', backward=map_, name=name)
    basepath = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(basepath, 'titanic_cleaned.csv'))
    df_orig = dtm.backward(df, inplace=False)
    df_orig.to_csv(os.path.join(basepath, 'titanic_original.csv'), index=False)
    pass