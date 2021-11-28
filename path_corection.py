import pickle
new_prefix = "/local/temporary/"

for txt in ["train", "val", "test"]:
    with open("/local/temporary/DATASET/"+txt+".data", 'rb') as annot:
        annot = pickle.load(annot)

    for image in annot:
        path = image["file_name"]
        #print(path)
        image["file_name"] = new_prefix + path
        #print(new_prefix + path)

    with open(txt+"-correct.data", 'wb') as file:
        pickle.dump(annot, file)
