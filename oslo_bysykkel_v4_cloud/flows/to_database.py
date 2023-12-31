import os
import pandas as pd

import sqlalchemy


url = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/2023/09.csv"
csv_name = f"{url[-11:-8]}_{url[-7:-5]}.csv"



def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    USERNAME = os.environ.get('DB_USER')
    PASSWORD = os.environ.get('DB_PASS')
    DATABASE = os.environ.get("DATABASE")
    unix_socket_path = os.environ[
        "INSTANCE_UNIX_SOCKET"
    ]


    pool = sqlalchemy.create_engine(
        sqlalchemy.engine.url.URL.create(
            drivername="postgresql+pg8000",
            username=USERNAME,
            password=PASSWORD,
            #host=HOST,
            #port=PORT,
            database=DATABASE,
            query={"unix_sock": f"{unix_socket_path}/.s.PGSQL.5432"},
        )     
    )

    return pool

def main():
    engine = connect_tcp_socket()
    df = pd.read_csv(url, nrows=3000)
    df.drop(["start_station_description", "end_station_description"], axis=1, inplace=True)
    # converting date columns to datetime
    df.head(n=0).to_sql(con=engine, name=os.environ.get("DATABASE"), if_exists="append")
    df.started_at = pd.to_datetime(df.started_at)
    df.ended_at = pd.to_datetime(df.ended_at)
    df.started_at = df.started_at.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    df.ended_at = df.ended_at.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_sql(con=engine, name=os.environ.get("DATABASE"), if_exists="append")
    print("Sent 3000, rows to Postgres")


if __name__ == "__main__":
    main()


