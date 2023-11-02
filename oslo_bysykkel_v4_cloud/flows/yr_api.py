from datetime import datetime
from to_database import connect_tcp_socket
import requests
import pandas as pd


def weather_within_1_hour(api_input) -> list:
    """Requires the YR API raw input, and returns the type of weather and rain amount"""
    index = 0
    for count, d in enumerate(api_input['properties']['timeseries']):
      if int(d['time'].split("T")[1][:2]) == datetime.now().hour + 2:
        index = count
        break
    type_of_weather = api_input["properties"]['timeseries'][index]['data']['next_1_hours']['summary']['symbol_code']
    rain_amount = api_input["properties"]['timeseries'][index]['data']['next_1_hours']['details']['precipitation_amount']
    temp = api_input["properties"]['timeseries'][index]['data']['instant']['details']['air_temperature']
    return [datetime.now().strftime("%Y-%m-%d"), type_of_weather, rain_amount, temp]


def get_weather_forecast(coord: list) -> dict:
    """Retrieves the raw API output"""
    lat = coord[0]
    lon = coord[1]
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"

    headers = {
        'User-Agent': 'MyApp (admin@admin.com)'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()

    else:
        raise ConnectionAbortedError("Failed to retrieve data, : {response.status_code}")

def lst_to_df(data: list) -> pd.DataFrame:
   """Converts the list to a DataFrame object"""
   return pd.DataFrame(data, index=['date', "weather", "rain_amount", "temperature"]).T

def data_to_gcp(table_name: str) -> None:
   """sends the data to the table name"""
   oslo_lat = 59.921
   oslo_lon = 10.745
   api_call = get_weather_forecast([oslo_lat, oslo_lon])
   api_filter = weather_within_1_hour(api_call)
   df = lst_to_df(api_filter)
   # connecting to gc postgres
   engine = connect_tcp_socket()
   # only running the code if 
   #df.head(n=0).to_sql(con=engine, name=table_name, if_exists="replace")
   df.to_sql(con=engine, name=table_name, if_exists="append")


if __name__ == "__main__":
    table_name = "yr_data"
    data_to_gcp(table_name)