import numpy as np
def load_numpy(file_path):
    with open(file_path,'rb') as file:
         return np.load(file)
    
def save_numpy(file_path,data):
     with open(file_path,'wb') as file:
          np.save(file,data)