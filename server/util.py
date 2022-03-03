import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, bedrooms, bathrooms):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    X = np.zeros(len(__data_columns))
    X[0] = bedrooms
    X[1] = bathrooms
    if loc_index >= 0:
        X[loc_index] = 1

    return round(__model.predict([X])[0], 2)


def load_saved_artifacts():
    # print('loading saved artifacts...start')
    global __data_columns
    global __locations
    global __model

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[2:]

    with open('./artifacts/nairobi_home_prices_model2.pickle', 'rb') as f:
        __model = pickle.load(f)

    print('Loading saved artifacts done')


def get_location_names():
    global __locations
    return __locations


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('kileleshwa nairobi, kileleshwa, nairobi', 3, 3))
    # print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))

    # stop:13
