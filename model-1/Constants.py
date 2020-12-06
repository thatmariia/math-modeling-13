NRIMAGES = 2
IMAGE_IDS = [651, 714] # lemon and onion
MANUAL_TEST_DATA = True
NRCHANNELS = 3
RESOLUTION = (int(192/2), int(144/2))
NRPIXELS = RESOLUTION[0] * RESOLUTION[1] * NRCHANNELS
NRTESTS = 0.0

def createLabels():
    labelDict = {}
    objectIDs = list(set([objectID for objectID in IMAGE_IDS]))
    for i in range (len(objectIDs)):
        labelDict[objectIDs[i]] = i
    return labelDict


LABEL_IDS = createLabels()

FOLDERNAMES = ["data/aloi_red4_col", "data/aloi_red4_ill", "data/aloi_red4_stereo", "data/aloi_red4_view"]
FILENAME_MAP = {
    "data/aloi_red4_col"     : ["_i{}.png".format(i) for i in range (110, 191, 10)] + ["_i{}.png".format(i) for i in range (210, 251, 20)],
    "data/aloi_red4_ill"     : ["_l{}c{}.png".format(i,j) for i in range(1, 9) for j in range(1,4)],
    "data/aloi_red4_stereo"  : ["_{}.png".format(i) for i in ["c", "l", "r"]],
    "data/aloi_red4_view"    : ["_r{}.png".format(i) for i in range(0, 356, 5)]
}

FOLDERNAME_TEST = "data/test_data"
TEST_MAP = {
    #FOLDERNAME_TEST + "/651/651_1.png" : 651,
        FOLDERNAME_TEST + "/714/714_1.png": 714,
        FOLDERNAME_TEST + "/714/714_2.png": 714
}

for i in range(1, 2692):
        TEST_MAP[FOLDERNAME_TEST + "/651/651_{}.jpg".format(i)] = 651