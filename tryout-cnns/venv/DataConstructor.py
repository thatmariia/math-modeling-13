from Constants import *

from PIL import Image
from numpy import asarray
import pandas as pd
import random



class DataConstructor:

    def __init__(self):

        self.df         = self.getEmptyDF()
        self.df_test    = self.getEmptyDF()
        self.df_train   = self.getEmptyDF()

    def construct(self):

        for folder in FOLDERNAMES:
            for objectID in range(1, NRIMAGES+1):

                if (objectID%10 == 0):
                    print(objectID)

                filenames = self.getFilenames(index=objectID, folder=folder)
                nrFiles = len(filenames)

                indexList = list(range(nrFiles))
                testIndices = random.sample(indexList, int(NRTESTS*nrFiles))

                for i in range(nrFiles):
                    fp = folder + "/{}/".format(objectID) + filenames[i]
                    im = self.transformImage(filepath=fp)
                    row = self.createRowDict(label=objectID, im=im, columns=self.df.columns)
                    self.df = self.df.append(row, ignore_index=True)

                    if i in testIndices:
                        self.df_test = self.df_test.append(row, ignore_index=True)
                    else:
                        self.df_train = self.df_train.append(row, ignore_index=True)


    def getEmptyDF(self):
        columns = ["label"] + ["pixel{}".format(i) for i in range(NRPIXELS)]
        return pd.DataFrame(columns=columns)

    def createRowDict(self, label, im, columns):
        rowDict = { columns[0] : label }
        for i in range(im.shape[0]):
            rowDict[columns[i+1]] = im[i]
        return rowDict

    def getFilenames(self, index, folder):
        fns = FILENAME_MAP[folder]
        return ["{}{}".format(index, fns[i]) for i in range(len(fns))]

        #return ["{}_i{}.png".format(index,i) for i in range (110, 191, 10)] + ["{}_i{}.png".format(index,i) for i in range (210, 251, 20)]

    def transformImage(self, filepath):
        image = Image.open(filepath)
        image = image.resize((RESOLUTION[0], RESOLUTION[1]))
        data = asarray(image, dtype="float")
        return data.flatten()
