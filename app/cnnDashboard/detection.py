from darkflow.defaults import argHandler #Import the default arguments
from darkflow.net.build import TFNet
import os
import base64
import shutil

def setup_image_directory(directoryName="sample_img", outputName='out'):
    folder = os.path.join(os.getcwd(), "app", "cnnDashboard", directoryName)
    print(f"Empting the following directory {folder}")
    # Empty that directory
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            print(filename)
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"[INFO] {filename} removed")
            except Exception as e:
                print('[ERROR] Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        print(f"[ERROR] setup {folder} does not exist")
        
    # Creating the outfile
    access_rights = 0o777
    path = os.path.join(folder, outputName)
    print(f"Saving output file at: {path}")
    try:
        os.mkdir(path, access_rights)
    except OSError as e:
        print ("[ERROR] Creation of the directory %s failed" % path)
        print(e)
    else:
        print(f"[INFO] created a new directory at {path}")

# save image in sample_img
def save_as_jpg(filename, b64String, directoryName="sample_img"):
    path = os.path.join(os.getcwd(),"app", "cnnDashboard", directoryName)
    print(f"Saving the new jpg file at : {path}")
    if os.path.isdir(path):
        filename = os.path.join(path, filename)
        print(f"[INFO] save file = {filename}")
        image = base64.b64decode(b64String)
        with open(filename, 'wb') as f:
            f.write(image)
    else:
        print(f"[ERROR] ./{directoryName} does not exist")    

# object detection parameters
def image_prediction():
    current_dir = os.getcwd()+"/app/cnnDashboard"
    FLAGS = {
        'imgdir': f'{current_dir}/sample_img/', 
        'binary': f'{current_dir}/bin/', 
        'config': f'{current_dir}/cfg/', 
        'backup': './ckpt/', 
        'threshold': -0.1, 
        'model': f'{current_dir}/cfg/yolo.cfg', 
        'load': f'{current_dir}/bin/yolo.weights', 
        'gpu': 0.0, 
        'batch': 16
        }
    tfnet = TFNet(FLAGS)
    tfnet.predict()

def read_file(filename,prefix,mainD="sample_img", outputD="out"):
    file = os.path.join(os.getcwd(), 'app', 'cnnDashboard', mainD, outputD, filename)
    with open(file, 'rb') as f:
        data = f.read()
    image = prefix+"base64,"+base64.b64encode(data).decode()
    return image