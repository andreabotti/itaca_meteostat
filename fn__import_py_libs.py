import streamlit as st
from meteostat import Stations, Hourly, Point
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import pydeck as pdk
import folium
from streamlit_folium import st_folium, folium_static
# from folium.plugins import ClickForMarker



#####
def create_page_header():
    st.markdown(
        """<style>.block-container {padding-top: 1rem; padding-bottom: 0rem; padding-left: 3rem; padding-right: 3rem;}</style>""",
        unsafe_allow_html=True)
    #
    # TOP CONTAINER
    TopColA, TopColB = st.columns([6,2])
    with TopColA:
        st.markdown("## Download weather data from Meteostat")
        # st.markdown("#### Analisi di dati meteorologici ITAliani per facilitare l'Adattamento ai Cambiamenti Climatici")
        st.caption('Developed by AB.S.RD - https://absrd.xyz/')
    #
    st.markdown('---')



# Function to display a map and capture click events
def display_map():
    # Initial map center and zoom level
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=37.7749,
        longitude=-122.4194,
        zoom=11,
        pitch=0,
    )

    # PyDeck layer for displaying map (no data in this example, just base map)
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=pd.DataFrame({
            'lat': [],
            'lon': [],
        }),
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=200,
    )

    # Render map with PyDeck
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[layer],
    ))

    # Handling map click events
    map_click_info = st.session_state.get("map_click_info", None)
    if map_click_info:
        lat, lon = map_click_info["coordinates"]
        st.write(f"Latitude: {lat}, Longitude: {lon}")
    else:
        st.write("Click on the map to get the latitude and longitude.")

# Callback to capture click events
def map_click_handler(event_info):
    st.session_state["map_click_info"] = event_info








#####
st.set_page_config(page_title="Meteostat App",   page_icon=':mostly_sunny:', layout="wide")
create_page_header()



st.markdown('##### Choose a location')
col1, col2, col_table = st.columns([2,1,8])

coords_EETRA = [45.47657243381954, 9.218019976414713]
coords_SW = [51.532205, -0.099013]

with col1:
    m = folium.Map(location=coords_EETRA, zoom_start=16)

    folium.Marker(
        coords_SW,
        popup="Stanton Williams",
        ).add_to(m)
    folium.Marker(
        coords_EETRA,
        popup="EETRA",
        tooltip="EETRA",
        ).add_to(m)

    st_data = st_folium(m, height=300, use_container_width=True)

try:
    lat = st_data['last_clicked']['lat']
    lon = st_data['last_clicked']['lng']
except:
    lat = 0.1
    lon = 0.1
    col1.write("Click on the map to place a marker.")


if lat:
    set_lat = col2.number_input("Latitude:",  value=lat, format="%.4f")
    set_lon = col2.number_input("Longitude:", value=lon, format="%.4f")
    distance_threshold_km = col2.number_input("Max Distance (km):", min_value=0, value=30, step=5)
else:
    col2.write("Click on the map to place a marker.")


stations = Stations()
nearby_stations = stations.nearby(lat, lon).fetch()




def fetch_temperature_data(station_id, lat, lon):
    # Create Point for the specified latitude and longitude
    location = Point(lat, lon)

    # Fetching hourly data for the chosen location and date range
    data = Hourly(location, start_date, end_date)
    df0 = data.fetch()

    # Displaying the data
    if not df0.empty:
        df = df0['temp']

    return df




if not nearby_stations.empty:
    # Convert distance to kilometers (assuming distance is provided in meters)
    if 'distance' in nearby_stations.columns:
        nearby_stations['distance'] = nearby_stations['distance'] / 1000  # Convert m to km
        # Filter DataFrame based on distance threshold
        filtered_stations = nearby_stations[nearby_stations['distance'] <= distance_threshold_km]

        # APPLYING SOME TABLE FORMATTING
        df = filtered_stations
        df['distance'] = df['distance'].round(1)

        df.drop(['monthly_start', 'monthly_end'], axis=1, inplace=True)
        df.drop(['timezone'], axis=1, inplace=True)
    
        # Apply formatting to exclude minutes and seconds, keep the hour if it's not 00
        for col in ['hourly_start', 'hourly_end', 'daily_start', 'daily_end']: 
            df[col] = df[col].dt.strftime("%Y-%m-%d") + df[col].apply(lambda x: '' if x.hour == 0 else ' %H')
        filtered_stations = df


        # Displaying the filtered stations as a DataFrame in Streamlit
        with col_table.container():
            st.markdown(f'###### Weather stations found within {distance_threshold_km} km from location')
            st.dataframe(filtered_stations, use_container_width=True)

    else:
        st.error("Distance information is not available in the stations data.")
else:
    st.write("No nearby stations found.")



st.markdown('---')
st.markdown('##### Plot temperature data')

col_stats, col_table = st.columns([2,8])

with col_stats:
    col_stats1, col_stats2 = st.columns([1,1])

start_date =    col_stats1.date_input("Start date",    value=datetime(2024,1,1))
end_date =      col_stats2.date_input("End date",        value=datetime(2024,2,29))

# col_stats.markdown('##### Main Statistics')

# Example for selecting a station, assuming 'id' is a column in your DataFrame
chosen_station_ids = col_table.multiselect("Choose Station IDs:", filtered_stations.index.unique())

# Ensure both start and end dates are datetime.datetime objects at the start of the respective day
start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.min.time())
end_date = end_date + timedelta(days=1) - timedelta(seconds=1)      # Adding time to the end_date to include the entire day in the query




# if st.button("Fetch Hourly Data for Selected Station"):

# This line initializes a Plotly figure
fig = go.Figure()
temp_data_table = pd.DataFrame()



for station_id in chosen_station_ids:
    
    station_data = filtered_stations.loc[station_id]

    try:
        temp_data = fetch_temperature_data(
            station_id=station_data.index,
            lat=station_data.latitude,
            lon=station_data.longitude,
            ).to_frame()

        temp_data_table = pd.concat(
            [temp_data_table, temp_data.rename(columns={'temp':f'{station_id}__temp'})],
            axis=1,
            )

        # Create a trace for the current station's temperature data
        trace = go.Scatter(
            x=temp_data.index,  # Assuming the DataFrame index is the datetime
            y=temp_data['temp'],  # Assuming 'temp' column holds the temperature data
            mode='lines',  # Line plot
            line=dict(width=1),  # Black, thin, dotted line
            name=str(station_id)  # Use the station ID as the trace name
        )
        fig.add_trace(trace)

    except:
        f'Hourly data not available for {station_id}'


# Compute the hourly average across all stations (columns)
temp_data_table['AVG'] = temp_data_table.mean(axis=1)
trace = go.Scatter(
    x=temp_data_table.index,  # Assuming the DataFrame index is the datetime
    y=temp_data_table['AVG'],  # Assuming 'temp' column holds the temperature data
    mode='lines',  # Line plot
    name='AVG',  # Use the station ID as the trace name
    line=dict(color='black', width=3, dash='dot'),  # Black, thin, dotted line
    )
fig.add_trace(trace)


fig.update_layout(
title='Temperature Data by Station ID',
xaxis=dict(
    # title='Time',
    rangeselector=dict(
        buttons=list([
            dict(count=3, label="3d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=14, label="14d", step="day", stepmode="backward"),
            dict(step="all")
        ])
    ),
    type="date",
    rangeslider=dict(visible=True),
),
yaxis_title='Temperature (°C)',
legend_title='Station ID',
height=600,
# width=800,
margin=dict(l=10, r=10, t=100, b=10),
)


# Initialize a Plotly figure
fig_box = go.Figure()
df_plot = temp_data_table

# Add a box plot for each station column in the DataFrame
for col in df_plot.columns:
    # st.write(col)
    fig_box.add_trace(go.Box(y=df_plot[col], name=col))

# Customize the layout
fig_box.update_layout(
    title       = 'Temperature Distribution by Station and AVG',
    yaxis_title = 'Temperature (°C)',
    boxmode     = 'group'  # Group boxplots together for each category (station)
)




col_table, col_chart = st.columns([2,8])


col_table.plotly_chart(fig_box, use_container_width=True)

col_chart.plotly_chart(fig, use_container_width=True)

with col_chart:
    with st.expander('Tabular Data'):
        st.dataframe(temp_data_table,use_container_width=True)
