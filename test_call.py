from subprocess import call

if __name__ == "__main__":

    call(['python main_readtxt.py', 'imagelist.txt', 'class.out', '--contours=40'])
