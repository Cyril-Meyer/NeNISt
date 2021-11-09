import os


def get(default='/home/cyril/Documents/Data/'):
    data_folder = default
    name = os.uname()[1]
    # local machine, only 1 cuda gpu.
    if(name == 'lythandas'):
        data_folder = '/home/cyril/Documents/Data/'
    # gpu servers, 4 gpu per server.
    elif(name == 'lsiit-miv-gpu1' or name =='icube-images-gpu2'):
        data_folder = '/dataUser/miv/cyril.meyer/data/'
    elif 'hpc' in name:
        data_folder = '/b/home/miv/cmeyer/DATA/'
    
    return data_folder
