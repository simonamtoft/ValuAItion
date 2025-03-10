import json
import numpy as np
import pandas as pd
import geopandas as gpd
import dash_leaflet as dl
from dash import Dash, html, Input, Output, State
from src.model.ValueModel import ValueModel
from src.app.helpers import setup_input_fields, get_min_distance, get_coord_vectors

app = Dash(__name__)
model = ValueModel(run_id='65fdd6f4-4ab7-4a4d-b591-d51fa3f3890d')

# Read the shapefile for coastline
coastline_geojson = json.loads(gpd.read_file('data/ne_10m_coastline.shp').to_json())
coords_coast = get_coord_vectors(coastline_geojson, 10)
lakes_geojson = json.loads(gpd.read_file('data/ne_10m_lakes.shp').to_json())
coords_lakes = get_coord_vectors(lakes_geojson, 5)

# Copenhagen, (lat, lon) coordinates
initial_map_position = [55.676, 12.568]

input_fields, features = setup_input_fields()
map_component = dl.Map(
    id='map',
    center=initial_map_position,
    zoom=13,
    children=[
        dl.TileLayer(),
        dl.GeoJSON(data=coastline_geojson, id='coastline'),
        dl.Marker(position=initial_map_position, id='marker')],
    style={
        'width': '80%',
        'height': '50vh',
        'margin': 'auto',
        'display': 'block'}
)

app.layout = html.Div([
    html.H1("ValuAItion"),
    *input_fields,
    html.Button('Predict', id='predict-button'),
    map_component,
    html.Div(id='prediction-output'),
    html.P(id="click-output")
])


def get_state(feature: dict):
    if feature['type'] in ['number', 'category', 'boolean', 'date_year']:
        return State(feature['id'], 'value')
    else:
        return State(feature['id'], 'date')


@app.callback(
    [Output('marker', 'position'),
     Output('LAT', 'value'),
     Output('LNG', 'value'),
     Output('DISTANCE_COAST', 'value'),
     Output('DISTANCE_LAKE', 'value')],
    Input('map', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        init_pos = np.array([initial_map_position[0], initial_map_position[1]])
        dist_coast = get_min_distance(init_pos, coords_coast)
        dist_lake = get_min_distance(init_pos, coords_lakes)
        return initial_map_position, initial_map_position[0], initial_map_position[1], dist_coast, dist_lake
    new_pos = clickData['latlng']
    new_pos_arr = np.array([new_pos['lat'], new_pos['lng']])
    dist_coast = get_min_distance(new_pos_arr, coords_coast)
    dist_lake = get_min_distance(new_pos_arr, coords_lakes)
    return new_pos, new_pos['lat'], new_pos['lng'], dist_coast, dist_lake


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [get_state(feature) for feature in features]
)
def predict_price(n_clicks, *input_values):
    if n_clicks is None:
        return "Please enter data and click predict."

    # setup data for prediction
    input_data = pd.DataFrame([input_values], columns=[x['id'] for x in features])
    input_data = input_data.replace('None', np.nan).dropna(axis=1)

    # Assuming input_data needs to be preprocessed before prediction
    price = model.predict(input_data)[0]
    return f'Predicted Price: {price}'


if __name__ == '__main__':
    app.run_server(debug=True)
