import json

FIXED_PARAMETERS = {
}

DEFAULT_PARAMETERS = {
    "INDEX.PRIMARY": "./index",
    "INDEX.SECONDARY": "./secondary_index",

    "DATA.ROOT": "./dataset",
    "DATA.BUSINESS": "yelp_academic_dataset_business.json",
    "DATA.REVIEW": "yelp_academic_dataset_review.json",
    "DATA.SUBSET": "ID",

    "SEARCH.BUSINESS.INDEX_ELIMINATION_MAX_RESULTS" : 1000,
    "SEARCH.BUSINESS.MIN_LEN" : 3,
    "SEARCH.REVIEW.MIN_LEN" : 3,
    "SEARCH.REVIEW.FEATURE_WEIGHTS_USEFUL" : 1.0,
    "SEARCH.REVIEW.FEATURE_WEIGHTS_FUNNY" : 0.5,
    "SEARCH.REVIEW.FEATURE_WEIGHTS_COOL" : 0.2,
    "SEARCH.SUMMARY.COSINE_SIMILARITY_THRESH" : 0.9,
    "SEARCH.SUMMARY.MAX_REVIEWS" : 10000,
    
    "KMEANS.N_CLUSTERS": 50,
    "KMEANS.RANDOM_SEED": 42,
    "TSVD.N_COMPONENTS": 200,
    "TSVD.N_ITERATIONS": 20,
    "TSVD.RANDOM_SEED": 42,

    "SEARCH.APPLICATION.HISTORY_MAX_LEN" : 10,
    "SEARCH.APPLICATION.RESULT_MAX_WIDTH": 200
}

VALIDATION_ARGS = [
    ('INDEX.PRIMARY', str),
    ('INDEX.SECONDARY', str),
    ('DATA.ROOT', str),
    ("DATA.BUSINESS", str),
    ("DATA.REVIEW", str),
    ("DATA.SUBSET", str),
    ("SEARCH.BUSINESS.INDEX_ELIMINATION_MAX_RESULTS", int, 1000, 1000),
    ("SEARCH.BUSINESS.MIN_LEN", int, 1),
    ("SEARCH.REVIEW.MIN_LEN", int, 1),
    ("SEARCH.REVIEW.FEATURE_WEIGHTS_USEFUL", float, 0, 1),
    ("SEARCH.REVIEW.FEATURE_WEIGHTS_FUNNY", float, 0, 1),
    ("SEARCH.REVIEW.FEATURE_WEIGHTS_COOL", float, 0, 1),
    ("SEARCH.SUMMARY.COSINE_SIMILARITY_THRESH", float, 0, 1),
    ("SEARCH.SUMMARY.MAX_REVIEWS", int, 1, 10000),
    ("KMEANS.N_CLUSTERS", int, 1, 200),
    ("TSVD.N_COMPONENTS", int, 1),
    ("TSVD.N_ITERATIONS", int, 1),
    ("KMEANS.RANDOM_SEED", int),
    ("TSVD.RANDOM_SEED", int),
    ("SEARCH.APPLICATION.HISTORY_MAX_LEN", int, 1, 1000),
    ("SEARCH.APPLICATION.RESULT_MAX_WIDTH", int, 40)
]

def validate_key(_dict, key):
    return key in _dict

def validate_range(val, lo, hi):
    if lo is None and hi is None:
        return True
    elif lo is None:
        return val <= hi
    elif hi is None:
        return lo <= val
    else:
        return lo <= val <= hi

def validate_int(_dict, key, lo=None, hi=None):
    if not validate_key(_dict, key):
        return False
    val = _dict.get(key)
    if type(val) != int:
        return False
    return validate_range(val, lo, hi)

def validate_float(_dict, key, lo=None, hi=None):
    if not validate_key(_dict, key):
        return False
    val = _dict.get(key)
    if not (type(val) == int or type(val) == float):
        return False
    return validate_range(val, lo, hi)

def validate_str(_dict, key):
    if not validate_key(_dict, key):
        return False
    val = _dict.get(key)
    if type(val) != str:
        return False
    if len(val.strip()) == 0:
        return False
    return True

def validate_parameters(parameters):
    for args in VALIDATION_ARGS:
        n = len(args)
        key, _type = args[0], args[1]
        if _type == str:
            # type validation only
            if not validate_str(parameters, key):
                print(f"Parameter {key} invalid.")
                return False
        else:
            val_func = validate_int if _type == int else validate_float
            # type validation only
            if n == 2:
                if not val_func(parameters, key, lo=None, hi=None):
                    print(f"Parameter {key} invalid.")
                    return False
            # lower bound
            elif n == 3:
                if not val_func(parameters, key, lo=args[2], hi=None):
                    print(f"Parameter {key} invalid.")
                    return False
            # upper or lower AND upper bound
            else:
                if not val_func(parameters, key, lo=args[2], hi=args[3]):
                    print(f"Parameter {key} invalid.")
                    return False
    return True

def default_parameters():
    parameters = dict(DEFAULT_PARAMETERS)
    parameters = add_fixed_parameters(parameters)
    return parameters

def add_fixed_parameters(parameters):
    for key in FIXED_PARAMETERS:
        parameters[key] = FIXED_PARAMETERS[key]
    return parameters

def get_parameters(path):
    try:
        with open(path, 'r') as file:
            modifiable_parameters = json.load(file)
            valid = validate_parameters(modifiable_parameters)
            parameters = add_fixed_parameters(modifiable_parameters)
            if valid:
                return parameters
            else:
                print("Invalid parameters found: Using default parameters")
                return default_parameters()
    except Exception as e:
        print(e)
        print("Error occured while attemping to load parameters: Using default parameters")
        return default_parameters()

