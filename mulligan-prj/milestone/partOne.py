from skimage.transform import resize

# Part 1

# if os.path.isfile('content/train.tar.gz'):
if os.path.isfile('train.tar.gz'):
  !ls la*
else:
  !gdown https://drive.google.com/open?id=1kSc5EtnpgZAUJIhV-fEh
    XwVt-48HjfHj
  !tar -xvf train.tar.gz
  
# get pic count from dataframe
numPics = len(df.index)
#numPics = 8000
#print(numPics)

X = np.empty((numPics,64,64,3))
cnt = 0
labels = []

# iterate over dataframe
for idx, row in df.iterrows():
  #print('train/'+row['id']+'.jpg')
  img = imread('train/'+row['id']+'.jpg')
  #print(img.shape)
  X[cnt] = resize(img,(64,64,3))
  cnt += 1
  print(cnt)
  #print('label is: ', row['breed'])
  labels.append(row['breed'])
#   if(cnt >= 8000):
#     break