from datetime import datetime
import pandas as pd
from to_database import connect_tcp_socket

def fetch(dataset_url: str):
    """Retrieving the data file from the link, reduced to 200 for demo purposes"""
    df = pd.read_csv(dataset_url, nrows=200)
    return df

def clean(df:pd.DataFrame) -> pd.DataFrame:
    """dropping start station and end station, and erasing some
    redundant information such as milliseconds, and UTC offset"""
    # dropping two text columns
    df.drop(["start_station_description", "end_station_description"], axis=1, inplace=True)
    # converting date columns to datetime
    df.started_at = pd.to_datetime(df.started_at)
    df.ended_at = pd.to_datetime(df.ended_at)
    df.started_at = df.started_at.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    df.ended_at = df.ended_at.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    return df

def etl_web_to_gcs_bicycle() -> None:
    """The main etl function, month input should be mm format"""
    month = datetime.now().month
    year = datetime.now().year
    dataset_url = f"https://data.urbansharing.com/oslobysykkel.no/trips/v1/{year}/{month}.csv"
    df = fetch(dataset_url)
    df_clean = clean(df)
    write_postgres(df_clean)


def write_postgres(df: pd.DataFrame) -> None:
    print("Hello")
    """Sending data to postgress database"""
    engine = connect_tcp_socket()
    # assumed that this is not the first job
    df.to_sql(con=engine, name='oslo_bicycle_db', if_exists="append")


if __name__ == "__main__":
    etl_web_to_gcs_bicycle()




