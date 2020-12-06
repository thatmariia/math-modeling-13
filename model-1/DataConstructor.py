from Constants import *

from PIL import Image
from numpy import asarray
import pandas as pd
import random

class DataConstructor:

    def __init__(self):

        self.df = self.getEmptyDF()
        self.df_test = self.getEmptyDF()

    def construct(self):
        self.constructTest()
        self.constructData()

    def constructData(self):

        for folder in FOLDERNAMES:
            for objectID in IMAGE_IDS:

                if (objectID%10 == 0):
                    print(objectID)

                filenames = self.getFilenames(index=objectID, folder=folder)

                for i in range(len(filenames)):
                    fp = folder + "/{}/".format(objectID) + filenames[i]
                    im = self.transformImage(filepath=fp)
                    row = self.createRowDict(label=LABEL_IDS[objectID], im=im, columns=self.df.columns)
                    self.df = self.df.append(row, ignore_index=True)

    def constructTest(self):

        for (fp, objectID) in TEST_MAP.items():
            try:
                im = self.transformImage(filepath=fp)
                row = self.createRowDict(label=LABEL_IDS[objectID], im=im, columns=self.df_test.columns)
                if MANUAL_TEST_DATA:
                    self.df_test = self.df_test.append(row, ignore_index=True)
                else:
                    self.df = self.df.append (row, ignore_index=True)
            except:
                print("failure ", fp)
                pass

    def getEmptyDF(self):
        columns = ["label"] + ["pixel{}".format(i) for i in range(NRPIXELS)]
        return pd.DataFrame(columns=columns)

    def createRowDict(self, label, im, columns):
        rowDict = { columns[0] : label }
        for i in range(len(im)):
            rowDict[columns[i+1]] = im[i]
        return rowDict

    def getFilenames(self, index, folder):
        fns = FILENAME_MAP[folder]
        return ["{}{}".format(index, fns[i]) for i in range(len(fns))]

    def transformImage(self, filepath):
        image = Image.open(filepath)
        image = image.resize((RESOLUTION[0], RESOLUTION[1]), Image.ANTIALIAS)
        data = asarray(image, dtype="float")
        d0 = [data[j][i][0] for i in range(data.shape[1]) for j in range(data.shape[0])]
        d1 = [data[j][i][1] for i in range(data.shape[1]) for j in range(data.shape[0])]
        d2 = [data[j][i][2] for i in range(data.shape[1]) for j in range(data.shape[0])]
        return d0 + d1 + d2
