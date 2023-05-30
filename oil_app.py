# Import necessary packages
import numpy as np
import dash
from dash import html, dcc, Input, Output, dash_table
import plotly.graph_objs as go
import pandas as pd
import geopandas as gpd
import plotly.express as px

# Loading the data from a pickle file
data = pd.read_pickle(r'data_oil.pickle')

# Calculating the 'width' column based on 'area' and 'length'
data['width'] = 4 * data.area / (np.pi * data.length)

# Generating the 'registry_nr' column with values ranging from 1 to the length of the data
data['registry_nr'] = [i for i in range(1, len(data) + 1)]

# Converting the 'date' column to datetime format
data['date'] = pd.to_datetime(data.date)

# Extracting the 'year' from the 'date' column
data['year'] = [date.year for date in data.date]

# Moving the 'year' and 'registry_nr' columns to the front of the DataFrame
data.insert(0, 'year', data.pop('year'))
data.insert(1, 'registry_nr', data.pop('registry_nr'))

# Load harbors and aerial stations

stations = pd.read_pickle(r'stations.pickle')

# translate
translation_dict = {
    'Pista/Porto': 'Runway/Port',
    'Local': 'Location',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'Veículo não Tripulado': 'Unmanned Vehicle',
    'lat': 'lat',
    'lon': 'lon',
    'usv_uav_autonomy_km': 'usv_uav_autonomy_km'
}

stations = stations.rename(columns=translation_dict)

translation_dict = {
    'Aéreo': 'Aerial',
    'Superfície': 'Surface'
}

stations['Unmanned Vehicle'] = stations['Unmanned Vehicle'].map(translation_dict)


# define color

stations['colors'] = ['red' if vehicle=='Aerial' else 'blue' for vehicle in stations['Unmanned Vehicle']]

import numpy as np
import pandas as pd
import plotly.express as px

from draw_circle_on_map import draw_circle_on_map


# Create a dictionary of month numbers and names
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# Get the minimum and maximum dates in the dataset
min_date = min(data['date'])
max_date = max(data['date'])

# Create a list of dates to use as the marks for the slider
dates = pd.date_range(start=min_date, end=max_date, freq='MS').tolist()

# Create a list of month numbers corresponding to the dates
month_numbers = [date.month for date in dates]

# Create a list of month names corresponding to the month numbers
month_names_list = [month_names[num] for num in month_numbers]

# Create a dictionary of marks for the slider
marks = {num: name for num, name in zip(month_numbers, month_names_list)}

# Get list of unique years and months in the dataset
unique_years = data['date'].dt.year.unique()

external_stylesheets = ['style.css']

# Create Dash app
app = dash.Dash(__name__, external_stylesheets = external_stylesheets
                )
server = app.server

app.title = 'Portuguese Potential Oil Spills'

# Define app layout
app.layout = html.Div(children=[
    # Page title
    html.H1('Potential_oil_spills',style={'text-align': 'center'}),

    # Tabs
    dcc.Tabs(id='tabs', value='map_tab', children=[
        # Tab for map
        dcc.Tab(label='Map', value='map_tab', children=[
            # Radio button to select area or length
            html.H4('Choose between circles representing the Area of the oil spill or representing the ratio between the real Area and the Area assuming that oil spill is an ellipse'),
            dcc.RadioItems(
                id='area_ratio',
                options=[{'label': 'Area', 'value': 'area'},
                         {'label': 'ratio', 'value': 'ratio'}],
                value='area',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            ),

            # Slider to select month and year range
            html.H3('Choose month and year ranges'),
            html.Div([
                dcc.RangeSlider(
                    id='year_slider',
                    min=data['date'].dt.year.min(),
                    max=data['date'].dt.year.max(),
                    value=[data['date'].dt.year.min(), data['date'].dt.year.max()],
                    marks={str(year): str(year) for year in data['date'].dt.year.unique()},
                    step=None,
                    vertical=True,
                    className='black-slider'
                ),
                dcc.RangeSlider(
                    id='month_slider',
                    min=min(month_numbers),
                    max=max(month_numbers),
                    value=[min(month_numbers), max(month_numbers)],
                    marks=marks,
                    step=None,
                    vertical=True,
                    className='black-slider',
                ),
                # Map to display oil spill locations
                html.Div([html.H3('Location of oil spill allerts'),
                          html.H4('Choose between static map and animated map by year'),
                          dcc.RadioItems(id='toggle-animation',
                                         options=[
                                                  {'label': 'None', 'value': 'None'},
                                                  {'label': 'Year', 'value': 'year'}], value='None',
                                         labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                                         ),
                          html.P('In this graph you may choose a set of points to see the data table bellow!'),
                          dcc.Graph(id='oil_map', style={'height': '600px', 'width': '1200px'}),
                        ], style={"marginTop": "20px",'backgroundColor': '#90EE90'}),  
            ], style={'display': 'flex'}),
        ]),

        # Tab for table
        dcc.Tab(label='Table of selected data', value='stations_table', children=[
            # Table to display selected data
            dash_table.DataTable(
                id='selected_data_table',
                columns=[{"name": i, "id": i} for i in data.drop(columns = ['date']).columns],
                page_size=10
            )
        ]),

        # Tab for table of stations
        dcc.Tab(label='Stations', children=[
            # Table of stations
            html.Div([
                    # Map to display oil spill locations
                    html.Div([html.H3('Location of possible stations'),
                    html.P('In the table bellow one may choose the UAV and USV stations to be represented in the map'),
                    dcc.Graph(id='stations_map', style={'height': '500px', 'width': '1000px'})]),
                    dash_table.DataTable(
                                        id='stations_table',
                                        columns=[{"name":i, "id": i} for i in stations.columns],
                                        data = stations.to_dict('records'),
                                        row_selectable='multi',
                                        style_table={'height': '200px','overflowY': 'auto','whiteSpace': 'normal',
                                                     'width': '600px'},
                                        selected_rows=[1,4,7,9,11,16]
                                        ),
                    ]),
        ]),
    ])
])

# Define app callback
@app.callback(
    Output('oil_map', 'figure'),
    Input('area_ratio', 'value'),
    Input('year_slider', 'value'),
    Input('month_slider', 'value'),
    Input('toggle-animation', 'value')
)
def update_map(area_ratio, year_range, month_range, toggle):
    o_df = data[['area','lat','lon','length', 'date','ratio','year']]
    if toggle == 'None':
        toggle = None
    # Filter data based on year and month selection
    df = data[['area','lat','lon','length', 'date','ratio','year']]
    df = df[(df['date'].dt.year >= year_range[0]) & (df['date'].dt.year <= year_range[1])]
    df = df[(df['date'].dt.month >= month_range[0]) & (df['date'].dt.month <= month_range[1])]
    
    # Create scatter plot on map
    if max(data[area_ratio])>300:
        max_range = 200
        tickvals_ar_ra = [0,50,100,150,200]
        ticktext_ar_ra = ['0','50','100','150','>200']
    else:
        max_range = 1
        tickvals_ar_ra = [0,0.2,0.4,0.6,0.8,1]
        ticktext_ar_ra = ['0','0.2','0.4','0.6','0.8','>1']
    
    df['marker_size'] = df[area_ratio].copy()

    o_df['marker_size'] = o_df[area_ratio].copy()

    min_marker_size = o_df['marker_size'].min()

    max_marker_size = o_df['marker_size'].max()

    fig = px.scatter_mapbox(df,
                            lat='lat',
                            lon='lon',
                            hover_data=['lat','lon','area','ratio','length'],
                            size='marker_size',
                            color=area_ratio,
                            color_continuous_scale='jet',
                            animation_frame=toggle,
                            range_color=[0, max_range],
                            #opacity=0.5,
                            #size_max = max_marker_size/20,  # Specify the range of marker sizes
                            zoom=4.1,
                            center={'lat': 37.5, 'lon': -18.0})
    fig.update_layout(coloraxis_colorbar=dict(
        title=area_ratio,
        tickvals=tickvals_ar_ra,
        ticktext=ticktext_ar_ra,
    ))
       
    if toggle == 'year':
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 250
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(
        plot_bgcolor='#90EE90',
        paper_bgcolor='#90EE90'
    )
    return fig


# @app.callback(
#     Output('selected_data_table', 'data'),
#     Input('oil_map', 'selectedData'),
#     Input('selected_data_table', 'data')
# )
# def update_table(selectedData, selected_data_table):
#     selected_data = []
#     if selectedData:
#         points = selectedData['points']
#         selected_indices = [point['pointIndex'] for point in points]
#         selected_df = data.iloc[selected_indices]
#     else:
#         selected_df = data.copy()
#     selected_df = selected_df.drop(columns=['date'])
#     selected_data = selected_df.to_dict('records')
#     return selected_data

@app.callback(
    Output('selected_data_table', 'data'),
    Input('month_slider', 'value'),
    Input('year_slider', 'value'),
    Input('oil_map', 'selectedData')
)
def update_table(month_range, year_range,selectedData):
    if selectedData:
        selected_indices = [point['pointIndex'] for point in selectedData['points']]
        selected_df = data.iloc[selected_indices].copy()
    else:
        selected_df = data.copy()
    selected_data = selected_df[(selected_df['date'].dt.month >= month_range[0]) & (selected_df['date'].dt.month <= month_range[1]) &
                         (selected_df['date'].dt.year >= year_range[0]) & (selected_df['date'].dt.year <= year_range[1])]
    selected_data = selected_data.to_dict('records')
    return selected_data







@app.callback(
    Output('stations_map', 'figure'),
    Input('stations_table', 'data'),
    Input('stations_table', 'selected_rows')
)
def update_stations_map(data_s, selected_rows):
    # Create a dataframe with the selected rows
    selected_data = pd.DataFrame(data_s).iloc[selected_rows, :]

    # Create an empty map figure
    fig = px.scatter_mapbox(selected_data, lat='lat', lon='lon', zoom=3.5, height=600,color='Veículo não Tripulado',color_discrete_sequence=['red','blue'])
    fig.update_layout(mapbox_style="open-street-map")
    

    #tickvals_ar_ra = [0,50,100,150,200]
    #ticktext_ar_ra = ['0','50','100','150','>200']
    fig2 = px.scatter_mapbox(data,
                            lat='lat',
                            lon='lon',
                            hover_data=['lat','lon','area','ratio','length'],
                            #size='area',
                            #color='area',
                            #color_continuous_scale='jet',
                            #range_color=[0, 200],
                            #opacity=0.5,
                            zoom=4.1,
                            center={'lat': 37.5, 'lon': -18.0})
    # Loop through each row of selected_data and draw a circle on the map for each row
    
    for i, row in selected_data.iterrows():
        circle_fig = draw_circle_on_map(row['lat'], row['lon'], row['usv_uav_autonomy_km'], row['colors'])
        fig.add_trace(circle_fig.data[0])
    ''' fig = px.scatter_mapbox(selected_data,
                            lat='lat',
                            lon='lon',
                            zoom=4, height=600) '''
    fig.update_layout(mapbox_style="open-street-map")
    fig.add_trace(fig2.data[0])
    fig.update_layout(coloraxis_showscale=False)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

