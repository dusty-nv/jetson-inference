"""coco2kitti.py: Converts MS COCO annotation files to
                  Kitti format bounding box label files
__author__ = "Jon Barker"
"""

import os
from pycocotools.coco import COCO

def coco2kitti(catNms, annFile):

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # Create an index for the category names
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']

    for img in coco.imgs:

        # Get all annotation IDs for the image
        catIds = coco.getCatIds(catNms=catNms)
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)

        # If there are annotations, create a label file
        if len(annIds) > 0:
            # Get image filename
            img_fname = coco.imgs[img]['file_name']
            # open text file
            with open('./labels/' + img_fname.split('.')[0] + '.txt','w') as label_file:
                anns = coco.loadAnns(annIds)
                for a in anns:
                    bbox = a['bbox']
                    # Convert COCO bbox coords to Kitti ones
                    bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    bbox = [str(b) for b in bbox]
                    catname = cat_idx[a['category_id']]
                    # Format line in label file
                    # Note: all whitespace will be removed from class names
                    out_str = [catname.replace(" ","")
                               + ' ' + ' '.join(['0']*3)
                               + ' ' + ' '.join([b for b in bbox])
                               + ' ' + ' '.join(['0']*8)
                               +'\n']
                    label_file.write(out_str[0])

if __name__ == '__main__':

    # These settings assume this script is in the annotations directory
    dataDir = '..'
    dataType = 'train2014'
    annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)

    # If this list is populated then label files will only be produced
    # for images containing the listed classes and only the listed classes
    # will be in the label file
    # EXAMPLE:
    #catNms = ['person', 'dog', 'skateboard']
    catNms = []

    # Check if a labels file exists and, if not, make one
    # If it exists already, exit to avoid overwriting
    if os.path.isdir('./labels'):
        print('Labels folder already exists - exiting to prevent badness')
    else:
        os.mkdir('./labels')
        coco2kitti(catNms, annFile)
