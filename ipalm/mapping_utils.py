


material_all_str = ["brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass", "hair", "leather",
              "metal", "mirror", "other", "painted", "paper", "plastic", "polishedstone",
              "skin", "sky", "stone", "tile", "wallpaper", "water", "wood"]

material_ipalm_ignore = ["mirror", "sky", "skin", "leather", "hair",
                        "painted", "brick", "carpet", "fabric",
                        "foliage", "food", "polishedstone", "stone", "tile",
                        "wallpaper"]

material_ipalm2raw = [2, 6, 9, 11, 13, 14, 21, 22]

material_raw2ipalm = {2: 0, 6: 1, 9: 2, 11: 3, 13: 4, 14: 5, 21: 6, 22: 7}

# {2: 'ceramic', 6: 'glass', 9: 'metal', 11: 'other', 13: 'paper', 14: 'plastic', 21: 'water', 22: 'wood'}
# \material_ipalm2raw/
#  \=>
material_ipalm_id2str = {0: 'ceramic', 1: 'glass', 2: 'metal', 3: 'other', 4: 'paper', 5: 'plastic', 6: 'water', 7: 'wood'}

# material_list
material_ipalm_list = ['ceramic', 'glass', 'metal', 'other', 'paper', 'plastic', 'water', 'wood']
# category_list
category_ipalm_list =['baking tray', 'ball', 'blender', 'bottle', 'bowl', 'box', 'can', 'chopping board', 'clamp',
                      'coffee maker', 'cylinder', 'dice', 'drill', 'food box', 'fork', 'fruit', 'glass', 'hammer',
                      'kettle', 'knife', 'lego', 'mug', 'pan', 'pen', 'pill', 'plate', 'pot', 'scissors', 'screwdriver',
                      'soda can', 'spatula', 'spoon', 'thermos', 'toaster', 'wineglass', 'wrench']

material_str2ipalm_id = {'ceramic': 0, 'glass': 1, 'metal': 2, 'other': 3, 'paper': 4, 'plastic': 5, 'water': 6, 'wood': 7}

material_str2raw_id = {'brick': 0, 'carpet': 1, 'ceramic': 2, 'fabric': 3, 'foliage': 4, 'food': 5, 'glass': 6, 'hair': 7, 'leather': 8,
 'metal': 9, 'mirror': 10, 'other': 11, 'painted': 12, 'paper': 13, 'plastic': 14, 'polishedstone': 15, 'skin': 16,
 'sky': 17, 'stone': 18, 'tile': 19, 'wallpaper': 20, 'water': 21, 'wood': 22}
# print(material_str2raw_id)
material_raw_id2str = {0: 'brick', 1: 'carpet', 2: 'ceramic', 3: 'fabric', 4: 'foliage', 5: 'food', 6: 'glass', 7: 'hair', 8: 'leather',
 9: 'metal', 10: 'mirror', 11: 'other', 12: 'painted', 13: 'paper', 14: 'plastic', 15: 'polishedstone', 16: 'skin',
 17: 'sky', 18: 'stone', 19: 'tile', 20: 'wallpaper', 21: 'water', 22: 'wood'}
# print(material_raw_id2str)

# material_mobile2own = {"plastic": "hard/soft plastic/foam"}  # ???


## EXTRA MAPPING:
real_material_to_ipalm_material = {"hard plastic": "plastic", "soft plastic": "plastic", "foam": "other"}

shortened_id_to_str = {0: "baking tray", 1: "ball", 2: "blender", 3: "bottle", 4: "bowl", 5: "box", 6: "can", 7: "chopping board", 8: "clamp", 9: "coffee maker", 10: "cylinder", 11: "dice", 12: "drill", 13: "food box", 14: "fork", 15: "fruit", 16: "glass", 17: "hammer", 18: "kettle", 19: "knife", 20: "lego", 21: "mug", 22: "pan", 23: "pen", 24: "pill", 25: "plate", 26: "pot", 27: "scissors", 28: "screwdriver", 29: "soda can", 30: "spatula", 31: "spoon", 32: "thermos", 33: "toaster", 34: "wineglass", 35: "wrench"}


str_to_shortened_id = {'baking tray': 0, 'ball': 1, 'blender': 2, 'bottle': 3, 'bowl': 4, 'box': 5, 'can': 6, 'chopping board': 7, 'clamp': 8, 'coffee maker': 9, 'cylinder': 10, 'dice': 11, 'drill': 12, 'food box': 13, 'fork': 14, 'fruit': 15, 'glass': 16, 'hammer': 17, 'kettle': 18, 'knife': 19, 'lego': 20, 'mug': 21, 'pan': 22, 'pen': 23, 'pill': 24, 'plate': 25, 'pot': 26, 'scissors': 27, 'screwdriver': 28, 'soda can': 29, 'spatula': 30, 'spoon': 31, 'thermos': 32, 'toaster': 33, 'wineglass': 34, 'wrench': 35}


raw_id_to_catstr = {0: "baking tray", 1: "baking tray", 2: "ball", 3: "ball", 4: "ball", 5: "ball", 6: "blender", 7: "bottle", 8: "bowl", 9: "bowl", 10: "bowl", 11: "bowl", 12: "box", 13: "box", 14: "box", 15: "box", 16: "can", 17: "chopping board", 18: "chopping board", 19: "chopping board", 20: "clamp", 21: "coffee maker", 22: "coffee maker", 23: "cylinder", 24: "dice", 25: "dice", 26: "drill", 27: "food box", 28: "fork", 29: "fork", 30: "fork", 32: "fruit", 31: "fruit", 33: "glass", 34: "glass", 35: "hammer", 36: "kettle", 37: "kettle", 40: "knife", 38: "knife", 39: "knife", 41: "lego", 42: "mug", 43: "mug", 44: "mug", 45: "pan", 46: "pan", 47: "pen", 48: "pill", 49: "plate", 50: "plate", 51: "pot", 52: "scissors", 53: "screwdriver", 54: "soda can", 55: "spatula", 56: "spoon", 57: "spoon", 58: "spoon", 59: "thermos", 60: "thermos", 61: "toaster", 62: "toaster", 64: "wineglass", 63: "wineglass", 65: "wrench"}
raw_id_to_shortened_id = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4, 10: 4, 11: 4, 12: 5, 13: 5, 14: 5, 15: 5, 16: 6, 17: 7, 18: 7, 19: 7, 20: 8, 21: 9, 22: 9, 23: 10, 24: 11, 25: 11, 26: 12, 27: 13, 28: 14, 29: 14, 30: 14, 32: 15, 31: 15, 33: 16, 34: 16, 35: 17, 36: 18, 37: 18, 40: 19, 38: 19, 39: 19, 41: 20, 42: 21, 43: 21, 44: 21, 45: 22, 46: 22, 47: 23, 48: 24, 49: 25, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 31, 58: 31, 59: 32, 60: 32, 61: 33, 62: 33, 64: 34, 63: 34, 65: 35}

raw_id_to_rawmatstr = {0: "metal", 1: "soft plastic", 2: "glass", 3: "hard plastic", 4: "other", 5: "soft plastic", 6: "other", 7: "soft plastic", 8: "ceramic", 9: "glass", 10: "hard plastic", 11: "metal", 12: "foam", 13: "hard plastic", 14: "paper", 15: "wood",  "16": "can - metal", 17: "hard plastic", 18: "soft plastic", 19: "wood", 20: "hard plastic", 21: "metal", 22: "other", 23: "foam", 24: "foam", 25: "soft plastic", 26: "hard plastic", 27: "soft plastic", 28: "hard plastic", 29: "metal", 30: "other", 31: "hard plastic", 32: "soft plastic", 33: "glass", 34: "hard plastic", 35: "other", 36: "hard plastic", 37: "metal", 38: "hard plastic", 39: "metal", 40: "other", 41: "hard plastic", 42: "ceramic", 43: "hard plastic", 44: "metal", 45: "metal", 46: "other", 47: "hard plastic", 48: "foam", 49: "ceramic", 50: "metal", 51: "metal", 52: "other", 53: "other", 54: "metal", 55: "hard plastic", 56: "metal", 57: "hard plastic", 58: "other", 59: "hard plastic", 60: "metal", 61: "metal", 62: "other", 63: "glass", 64: "hard plastic", 65: "metal"}
raw_id_to_matstr = {0: "metal", 1: "plastic", 2: "glass", 3: "plastic", 4: "other", 5: "plastic", 6: "other", 7: "plastic", 8: "ceramic", 9: "glass", 10: "plastic", 11: "metal", 12: "other", 13: "plastic", 14: "paper", 15: "wood",  16: "metal", 17: "plastic", 18: "plastic", 19: "wood", 20: "plastic", 21: "metal", 22: "other", 23: "other", 24: "other", 25: "plastic", 26: "plastic", 27: "plastic", 28: "plastic", 29: "metal", 30: "other", 31: "plastic", 32: "plastic", 33: "glass", 34: "plastic", 35: "other", 36: "plastic", 37: "metal", 38: "plastic", 39: "metal", 40: "other", 41: "plastic", 42: "ceramic", 43: "plastic", 44: "metal", 45: "metal", 46: "other", 47: "plastic", 48: "other", 49: "ceramic", 50: "metal", 51: "metal", 52: "other", 53: "other", 54: "metal", 55: "plastic", 56: "metal", 57: "plastic", 58: "other", 59: "plastic", 60: "metal", 61: "metal", 62: "other", 63: "glass", 64: "plastic", 65: "metal"}




