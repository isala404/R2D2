import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors.nearest_centroid import NearestCentroid

def generateSamples():
    f = open('data.csv', 'w')
    f.write(
        'id,move,line1_1,line1_2,line1_3,line1_4,line1_5,line2_1,line2_2,line2_3,line2_4,line2_5,line3_1,line3_2,line3_3,line3_4,line3_5,line4_1,line4_2,line4_3,line4_4,line4_5,line5_1,line5_2,line5_3,line5_4,line5_5\n')

    def argMax(*arrays):
        bigArray = ""
        for array in arrays:
            for i in array:
                bigArray += str(i) + ','

        return bigArray[:-1] + '\n'

    def value():
        rand = random.randint(0, 5)
        if rand:
            return 0
        else:
            return 1

    for i in range(1000):
        line1 = [value(), value(), value(), value(), value()]
        line2 = [value(), value(), value(), value(), value()]
        line3 = [value(), value(), value(), value(), value()]
        line4 = [value(), value(), value(), value(), value()]
        line5 = [value(), value(), value(), value(), value()]

        move = str(random.randint(0, 2))

        write = argMax(line1, line2, line3, line4, line5)
        f.write(str(i) + ',' + move + ',' + write)

        print(i)

    f.close()

def train():
    df = pd.read_csv('data.csv')
    df.drop(['id'], 1, inplace=True)
    X = np.array(df.drop(['move'], axis=1))
    y = np.array(df['move'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = NearestCentroid(metric='euclidean', shrink_threshold=None)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    print(accuracy)
    example_measures = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    example_measures = example_measures.reshape(1, -1)
    prediction = clf.predict(example_measures)
    print(prediction)

train()
