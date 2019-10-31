import pickle
import os

'''
 @params
 file - filename
 data - the data you want to store in array format
 directory - the directory you want store the file in, only 1 directory, you may choose this to be a path
'''


def write_pickle_file(file, data, directory=None):
    #     directory = "picklefiles"
    if directory != None:
        mypath = f"./{directory}/{file}"
    else:
        mypath = f"./{file}"

    if(not(os.path.exists(mypath))):
        if directory != None:
            try:
                os.mkdir(directory)
            except:
                pass

    with open(mypath, 'wb') as f:
        pickle.dump(tuple(i for i in data), f)


'''
returns pickle variables as tuples
'''


def read_pickle_file(file, directory=None):
    if directory != None:
        mypath = f"./{directory}/{file}"
    else:
        mypath = f"./{file}"

    try:
        with open(mypath, 'rb') as f:
            arr = pickle.load(f)
    except:
        print("path is invalid")

    tup = tuple(i for i in arr)
    return tup
