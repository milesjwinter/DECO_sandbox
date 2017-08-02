#!/usr/bin/env python

#==============================================================================
# File Name : make_event_list.py
# Description :
# Creation Date : 06-10-2016
# Last Modified : Fri 10 Jun 2016 08:51:28 PM CDT
# Created By : Matt Meehan
#==============================================================================

import glob

if __name__ == "__main__":

    types = ['track', 'worm', 'spot']
    #types = ['track']
    dir = 'sortedEvents/'
    f = open ('%s_list.txt' % 'image', 'w')
    #dir  = 'sortedEvents/'
    for type in types:
        indir = '%s%s/' % (dir,type)
        print indir
        #with open ('%s_list.txt' % 'image', 'w') as f:
        for file in glob.iglob('%s*.jpg' % indir):
            f.write(file)
            f.write('\n')



