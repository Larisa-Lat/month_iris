import pickle

def predict(data):
    with open('model_pkl' , 'rb') as f:
        lr = pickle.load(f)
    return lr.predict(data)
