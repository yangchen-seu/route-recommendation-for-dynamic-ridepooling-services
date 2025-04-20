import os
import shutil
files = os.listdir('data/ODs/')
files = ['OD_update location1.csv']
for file in files:
    print(file)
    shutil.copyfile('data/ODs/' + file,'data/OD.csv')

    print('generate pickle')
    cmd = 'python generate_pickle.py'
    os.system(cmd)

    print('shortest path')
    cmd = 'python parallel_shortest_path_and_ego_graph.py'
    os.system(cmd)

    print('matching pairs')
    cmd = 'python parallel_searching_of_matching_pairs.py'
    os.system(cmd)

    print('predict')
    cmd = 'python predict.py'
    os.system(cmd)

    os.rename('result/predict_result.csv', 'result/predict_result_'+file)
# cmd = 'python .\generate_pickle.py'
# os.system(cmd)

# cmd = 'python .\parallel_shortest_path_and_ego_graph.py'
# os.system(cmd)

# cmd = 'python .\parallel_searching_of_matching_pairs.py'
# os.system(cmd)

# cmd = 'python .\predict.py'
# os.system(cmd)
