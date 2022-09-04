import pandas as pd 
import numpy as np 

class data_an():
    def __init__(self,data):
        self.data = data

    def rule4(self):
        if(self.data[self.data.columns[0]] == self.data[self.data.columns[0]][0]).all():
            self.data = self.data.drop([self.data.columns[0]],axis=1)
            return self.data
        else:
            return self.data 
    
    def ano_rule(self):
        if((self.data[self.data.columns[0]].all()).lower() == "column qualifaction" and "column information" ):
            self.n = self.data[self.data.columns[0]]
            self.data = self.data.drop([self.data.columns[0]],axis = 1)
            self.data['info'] = self.n 
            return self.data 
        #rule 3
        elif (self.data[self.data.columns[0]] == self.data[self.data.columns[1]]): 
            self.data.rename(columns = {self.data.columns[0],'step'},inplace = True )
        elif (self.data[self.data.columns[0]].startswith('unnamed') and self.data.columns[1] in ['Paramter','ProcessAttributes']):
            self.data.rename(columns = {self.data.columns[0],'step'},inplace = True)
        else:
            return self.data 
    
    def ano_lastrule(self):
        if(self.data[self.data.columns[0]].isnull().all() == True):
            self.n = self.data[self.data.columns[0]]
            self.title = self.data.columns[0]
            self.data = self.data.drop([self.data.columns[0]],axis=1)
            self.data[self.title] = self.n 

            return self.data 
        
        else:
            return self.data

    
df = pd.read_excel("C:/Users/Vimal/Desktop/ReadMe/Code/.gitignore/merge data.xlsx")
x = data_an(df)
x.rule4()
x.ano_rule()
x.ano_lastrule()
