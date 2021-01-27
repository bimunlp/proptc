import urllib.request
import tarfile
import os

folder_path = os.path.dirname(os.path.realpath(__file__))
print('Begin downloading of dataset')

dataset = 'ptc-corpus.tgz'
server = 'https://propaganda.qcri.org/ptc/data/'

print('downloading', dataset)
url = server + dataset
dataset_path = os.path.join(folder_path, dataset)
urllib.request.urlretrieve(url, dataset_path)

print('extracting', dataset)
with tarfile.open(dataset_path, 'r') as tar_file:
    tar_file.extractall(folder_path)
os.remove(dataset_path)