import pandas as pd
import json
import random


# data =json.load(open('/faststorage/COCO2017/annotations/captions_train2017.json'))

# imgs = data['images']
# random.shuffle(imgs)
# data['images'] = imgs[:256]

# out_file = open("dataloader/test.json", "w") 
# json.dump(data, out_file) 
# out_file.close()

data =json.load(open("dataloader/test.json"))
print(data.keys())
imgs = data['images']
print(len(imgs))