def read_split_from_str(list_of_dict_str):
    """
    You are given a string that resembles the declaration of a list of dictionaries. 
    Without using the pandas package, write a function read_split_from_str to split the data into two lists, one for training and one for testing, with a 70:30 split between the training set and the testing set. 
    Note: Use numpy for random number generation if you need to.
    input: "[{'x': 0.0, 'y': 5.43}, {'x': 50.0, 'y': 102.78}, {'x': 100.0, 'y': 204.24}]"
    output -> list: [
    [{'x': 0.0, 'y': 5.43}, {'x': 50.0, 'y': 102.78}],
    [{'x': 100.0, 'y': 204.24}]
    ]

    """
    list_of_dict = eval(list_of_dict_str)
    split_point = len(list_of_dict)*7 // 10
    return [list_of_dict[:split_point], list_of_dict[split_point:]]
