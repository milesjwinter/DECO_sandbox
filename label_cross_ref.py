import numpy as np
import matplotlib.pyplot as plt


def get_numtype(x): #Returns number value assigned to each type
    return {'worm':0,'spot':1,'track':2,'ambig':3,'noise':4,'edge':5,'empty':6}[x]
vec_get_numtype = np.vectorize(get_numtype)



#load data
miles_list = np.loadtxt('miles_track_labels.txt',dtype='S')
raaha_list = np.loadtxt('raaha_track_labels.txt',dtype='S')

#get image names
miles_paths = np.array([r.split('/')[-1].split('.')[:-1] for r in miles_list[:,-1]],dtype=int)
raaha_paths = np.array([r.split('/')[-1].split('.')[:-1] for r in raaha_list[:,-1]],dtype=int)

#get image labels
miles_labels = vec_get_numtype(miles_list[:,2])
raaha_labels = vec_get_numtype(raaha_list[:,2])

#sort numerically by image name
miles_indices = np.argsort(miles_paths,axis=0)
miles_sorted_paths = miles_paths[miles_indices]
miles_sorted_labels = miles_labels[miles_indices]

sorted_xavg = np.array(miles_list[miles_indices,0])
sorted_yavg = np.array(miles_list[miles_indices,1])
sorted_labels = np.array(miles_list[miles_indices,2])
sorted_labeler = np.array(miles_list[miles_indices,3])
sorted_paths = np.array(miles_list[miles_indices,4])

raaha_indices = np.argsort(raaha_paths,axis=0)
raaha_sorted_paths = raaha_paths[raaha_indices]
raaha_sorted_labels = raaha_labels[raaha_indices]

print sorted_xavg[0][0]
print sorted_xavg[0,:]
label_file = 'compared_labels.txt'
f = open(label_file,'a')
for i in range(len(sorted_xavg)):
    if miles_sorted_labels[i]==raaha_sorted_labels[i]:
        f.write("{}\t{}\t{}\t{}\t{}".format(sorted_xavg[i][0], sorted_yavg[i][0], sorted_labels[i][0], sorted_labeler[i][0], sorted_paths[i][0]))
        #f.write("%s\t%s\t%s\t%s\t%s"%(sorted_xavg[i][0], sorted_yavg[i][0], sorted_labels[i][0], sorted_labeler[i][0], sorted_paths[i][0]))
        f.write("\n")
f.close()


