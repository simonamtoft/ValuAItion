import json
import numpy as np
from dash import dcc
from datetime import date


def setup_input_fields():
    with open('features.json', 'r') as f:
        features = json.load(f)

    # setup inputs
    input_fields = []
    for feature in features:
        name = feature['id'].replace('_', ' ').title()
        if feature['type'] == 'number':
            input_fields.append(dcc.Input(
                id=feature['id'],
                type='number',
                placeholder=f'Enter {name}'
            ))
        elif feature['type'] == 'date':
            input_fields.append(dcc.DatePickerSingle(
                id=feature['id'],
                display_format='MMM Do, YYYY',
                placeholder=name
            ))
        elif feature['type'] == 'date_year':
            input_fields.append(dcc.Input(
                id=feature['id'],
                type='number',
                min=1600,
                max=date.today().year,
                placeholder=name
            ))
        elif feature['type'] == 'category':
            input_fields.append(dcc.Dropdown(
                id=feature['id'],
                options=feature['values'],
                placeholder=f'Select {name}'
            ))
        elif feature['type'] == 'boolean':
            input_fields.append(dcc.RadioItems(
                id=feature['id'],
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ]
            ))
    return input_fields, features


def get_min_distance(p1, p2):
    # radius of earth in KM
    r = 6371

    # compute distance
    d_lat = p2[:, 0] - p1[0]
    d_lon = p2[:, 1] - p1[1]
    a = np.sin(d_lat / 2)**2 + np.cos(p1[0]) * np.cos(p2[:, 0]) * np.sin(d_lon / 2)**2
    p_dist = 2 * np.asin(np.sqrt(a)) * r

    # return minimum distance
    min_idx = np.argmin(p_dist)
    min_dist = p_dist[min_idx]
    return min_dist.item()


def get_coord_vectors(geojson: dict, n_interpolated_points: int):
    number_of_points = 10
    coords = []
    for x in geojson['features']:
        c = x['geometry']['coordinates']

        # todo: handle MultiPolygon better later
        #  - actually something goes wrong..
        if x['geometry']['type'] == 'MultiPolygon':
            c = c[0]

        # interpolate between points
        for i in range(len(c) - 1):
            xs = np.linspace(c[i][0], c[i + 1][0], number_of_points + 2)
            ys = np.linspace(c[i][1], c[i + 1][1], number_of_points + 2)
            coords_i = np.stack([xs, ys], axis=1).tolist()
            coords.extend(coords_i)
    coords = np.array(coords)
    # coords are in (lon, lat) format, so we need to swap them
    coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0].copy()
    return coords
