import json

SCENES_FILE_PATH = '../Clevr_dataset/scenes/CLEVR_val_scenes.json'

def splitByNumberOfObjects(scenes):
    file_pointers = []
    for i in range(1,11):
        file = open(f'../Clevr_dataset/datasetSplits/{scenes["info"]["split"]}_images_objectsNum{i}.txt', 'w')
        file_pointers.append(file)

    for i in range(15000):
        objectsNumber = len(scenes['scenes'][i]['objects'])
        file_pointers[objectsNumber-1].write(f'{scenes["scenes"][i]["image_filename"]}\n')


def splitByInterIntraObjects(scenes):
    inter_fp = open(f'../Clevr_dataset/datasetSplits/{scenes["info"]["split"]}_images_inter.txt', 'w')
    intra_fp = open(f'../Clevr_dataset/datasetSplits/{scenes["info"]["split"]}_images_intra.txt', 'w')

    for i in range(15000):
        objectsList = scenes['scenes'][i]['objects']
        objectSet = set()

        for obj in objectsList:
            objectSet.add(obj['shape'])

        if (len(objectSet) > 1):
            inter_fp.write(f'{scenes["scenes"][i]["image_filename"]}\n')
        else:
            intra_fp.write(f'{scenes["scenes"][i]["image_filename"]}\n')


if __name__ == "__main__":
    with open(SCENES_FILE_PATH, 'r') as file:
        scenes  = json.load(file)

        splitByNumberOfObjects(scenes)
        splitByInterIntraObjects(scenes)
