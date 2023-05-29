import numpy as np
import pandas as pd
import plotly.express as px

def draw_circle_on_map(lat, lon, radius_km, color):
    # Convert radius from km to degrees of longitude using Haversine formula
    R_earth = 6371  # Earth's radius in km
    dlon = radius_km / (R_earth * np.cos(np.radians(lat)))
    dlat = radius_km / R_earth
    
    # Construct a circle polygon
    bearings = np.arange(0, 361)
    circle_lon = lon + np.degrees(dlon) * np.cos(np.radians(bearings))
    circle_lat = lat + np.degrees(dlat) * np.sin(np.radians(bearings))
    circle_df = pd.DataFrame({'lat': circle_lat, 'lon': circle_lon})
    
    # Create a scatter mapbox plot with the circle as a filled polygon
    fig = px.scatter_mapbox(circle_df, lat='lat', lon='lon', 
                            mapbox_style='carto-positron', zoom=8, 
                            center={'lat': lat, 'lon': lon}, 
                            hover_data={'lat': True, 'lon': True}, 
                            color_discrete_sequence=[color])
    fig.update_traces(mode='lines+markers', marker=dict(size=3), line = dict(width=3))
    
    return fig