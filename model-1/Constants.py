NRIMAGES = 2
IMAGE_IDS = [651, 714] # lemon and onion
NRCHANNELS = 3
RESOLUTION = (int(144/4), int(192/4))
NRPIXELS = RESOLUTION[0] * RESOLUTION[1] * NRCHANNELS
NRTESTS = 0.0

FOLDERNAMES = ["data/aloi_red4_col", "data/aloi_red4_ill", "data/aloi_red4_stereo", "data/aloi_red4_view"]
FILENAME_MAP = {
    "data/aloi_red4_col"     : ["_i{}.png".format(i) for i in range (110, 191, 10)] + ["_i{}.png".format(i) for i in range (210, 251, 20)],
    "data/aloi_red4_ill"     : ["_l{}c{}.png".format(i,j) for i in range(1, 9) for j in range(1,4)],
    "data/aloi_red4_stereo"  : ["_{}.png".format(i) for i in ["c", "l", "r"]],
    "data/aloi_red4_view"    : ["_r{}.png".format(i) for i in range(0, 356, 5)]
}
