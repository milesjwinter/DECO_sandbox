import os
import glob

if __name__ == "__main__":

    types = ['track', 'worm', 'spot']
    #types = ['two_particles_radioactivity','ambiguous']
    #types = ['track']
    dir = './sortedEvents/'

    for type in types:
        indir = '%s%s/' % (dir,type)
        for file in glob.iglob('%s*.jpg' % indir):
            file_num = file[-17:-4]
            os.system('python New_plotBlobs.py %s -o %s ' % (file,file_num))
            

