import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Encoder:
    def __init__(self,cat_cols):
        self.cat_cols = list(cat_cols)
        self.enc = OneHotEncoder(handle_unknown = "ignore",sparse_output= False )
        self.is_fitted = False
    def fit(self,df):
        self.enc.fit(df[self.cat_cols])
        self.is_fitted = True
        return self 
        
    def transform(self,df):
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
            
      
        encoded = self.enc.transform(df[self.cat_cols])
        
        feature_names = self.enc.get_feature_names_out(self.cat_cols)
        print("len(feature_names):", len(feature_names))
        print("first few feature names:", feature_names[:10])
        
        encoded_df = pd.DataFrame(encoded,columns = feature_names,index = df.index)
        
        df_out = pd.concat([df.drop(columns = self.cat_cols),encoded_df],axis = 1)  

        return df_out
