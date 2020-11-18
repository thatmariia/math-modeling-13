NRIMAGES = 100
RESOLUTION = (int(144/4), int(192/4))
NRPIXELS = RESOLUTION[0] * RESOLUTION[1]
NRTESTS = 0.1

FOLDERNAMES = ["grey4-illcol", "grey4-illdir", "grey4-viewdir"]
FILENAME_MAP = {
    "grey4-illcol"  : ["_i{}.png".format(i) for i in range (110, 191, 10)] + ["_i{}.png".format(i) for i in range (210, 251, 20)],
    "grey4-illdir"  : ["_l{}c{}.png".format(i,j) for i in range(1, 9) for j in range(1,4)],
    "grey4-viewdir" : ["_r{}.png".format(i) for i in range(0, 356, 5)]
}

# 108 images for 1 object