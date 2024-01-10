import os
from glob import glob
import shutil

savepath = '/home/bob/RAIL/piccom1/'
picpath = '/home/bob/RAIL/RAIL_duibi_images/'

modelname = ['BBS', 'CAVER', 'CSEP', 'LENO', 'ICON', 'CLA', 'CIR', 'DRER', 'RD3D']
picitem = [6,7,57]

for i in modelname:
    data_path = os.path.join(picpath, i) + '/rail_362/'
    for j in picitem:
        item_path = data_path + str(j) + '.png'
        # img_item = os.path.join(item_path, '.png')
        savepath_1 = os.path.join(savepath, i) + '/rail_362/'
        if not os.path.exists(savepath_1):
            os.makedirs(savepath_1)
        shutil.copy(item_path, savepath_1)
