# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:02:43 2022

@author: deube
"""

# import all libraries needed
#import numpy as np
#import pandas as pd
import tensorflow as tf
#from sklearn.preprocessing import StandardScaler
#from sklearn.base import BaseEstimator, TransformerMixin
#import tensorflow as tf



class hockey_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            self.reg = tf.keras.models.load_model("hockeyModel2")
            
        
  
        def predict_one(self, secondH, secondA, absDiff, totalGoals, homeWinning, isTie):
            unscaled_inputs = [[secondH, secondA, absDiff, totalGoals, homeWinning, isTie]]
            #scaler = StandardScaler()
            #scaler.fit(self.scaler)
            #scaled_inputs = scaler.transform(unscaled_inputs)
            pred = self.reg.predict(unscaled_inputs)
            return pred
        
        def label_predict(self, result):
            labeledResults = {}
            labels = ['1 to 0',	'0 to 1','0 to 0',	'1 to 1', '2 to 0','0 to 2','2 to 1',	
                      '1 to 2',	'3 to 0','2 to 2',	'3 to 1' , '0 to 3', '1 to 3', '2 to 3',	
                      '4 to 0', '3 to 2', '4 to 1', '0 to 4', '1 to 4', '4 to 2',	'2 to 4', "Other"]
            for i in range(result.shape[1]):
                labeledResults[labels[i]] = result[0][i]
            
            return labeledResults
        
        def label_top(self, labeledResults, top = 5):
            if top > len(labeledResults):
                top = len(labeledResults)
            
            best = sorted(labeledResults, key=labeledResults.get, reverse=True)[:top]
            dictBest = {}
            
            for i in best:
                dictBest[i] = labeledResults[i]
            
            self.best = dictBest
            
            return dictBest
                
    
            
        def expected_value(self, odds1, odds2, odds3, odds4, odds5, bet):
            
            
                        
            outcome1 = list(self.best.values())[0]
            outcome2 = list(self.best.values())[1]
            outcome3 = list(self.best.values())[2]
            outcome4 = list(self.best.values())[3]
            outcome5 = list(self.best.values())[4]
            
               
            ev1 = outcome1 * (bet * odds1) + (1 - outcome1) * -bet
            ev2 = outcome1 * ((bet * odds1)-bet) + outcome2 * ((bet * odds2)-bet) + (1 - outcome1 - outcome2) * (-bet * 2)
            ev3 = outcome1 * ((bet * odds1)-(bet*2)) + outcome2 * ((bet * odds2)-(bet*2)) + outcome3 * ((bet * odds3)-(bet*2)) + (1 - outcome1 - outcome2 - outcome3) * (-bet * 3)
            ev4 = outcome1 * ((bet * odds1)-(bet*3)) + outcome2 * ((bet * odds2)-(bet*3)) + outcome3 * ((bet * odds3)-(bet*3)) + outcome4 * ((bet * odds4)-(bet*3)) + (1 - outcome1 - outcome2 - outcome3 - outcome4) * (-bet * 4)
            ev5 = outcome1 * ((bet * odds1)-(bet*4)) + outcome2 * ((bet * odds2)-(bet*4)) + outcome3 * ((bet * odds3)-(bet*4)) + outcome4 * ((bet * odds4)-(bet*4)) + outcome5 * ((bet * odds5)-(bet*4)) + (1 - outcome1 - outcome2 - outcome3 - outcome4 - outcome5) * (-bet * 5)
    
    
            return ev1, ev2, ev3, ev4, ev5


