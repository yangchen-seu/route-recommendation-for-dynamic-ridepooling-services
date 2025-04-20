import pickle

with open('tmp/shortest_distances.pickle','rb') as f:
    tmp = pickle.load(f)
    

print(tmp)