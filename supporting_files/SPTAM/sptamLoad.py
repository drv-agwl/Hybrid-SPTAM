import dill
from covisibility import CovisibilityGraph, GraphKeyFrame, GraphMapPoint, GraphMeasurement
#import pickle
if __name__ == '__main__':
    infile3 = open('RK','rb')
    graph = CovisibilityGraph()
    graph=dill.load(infile3)
    #print(graph)
    #print(graph.kfs)
    #print(graph.pts)
    print(graph.kfs_set)
    for s in graph.kfs_set:
        print(s.id)
        for s1 in (s.meas):
            print(s1)
    #print(graph.meas_lookup)
