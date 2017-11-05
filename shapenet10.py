

class_id_to_name = {
    "0": "cup",
    "1": "bottle",
#    "1": "bathtub",
#    "2": "bed",
#    "3": "chair",
#    "4": "desk",
#    "5": "dresser",
#    "6": "monitor",
#    "7": "night_stand",
#    "8": "sofa",
#    "9": "table",
#    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }

class_names = set(class_id_to_name.values())
class_values = set(class_id_to_name.keys())

class_name_to_id = dict((y,x) for x,y in class_id_to_name.items())

#print(dictionary.items()) #prints keys and values || dict_items([('1', 'cup'), ('2', 'bottle')])
#print(dictionary.keys()) #prints keys             || dict_keys(['1', '2']) 
#print(dictionary.values()) #prints values	   || dict_values(['cup', 'bottle'])
