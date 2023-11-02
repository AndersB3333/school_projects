Automated data pipelines using Google Cloud


Introduction
This project consists of ETL scripts in Python that communicates with a database in Google cloud. I've connected this with Google Cloud Jobs to automatically execute the scripts.

Requirements

Google Cloud Account
Docker
Python 3.9
SQL Alchemy
pg8000

Google Cloud Services Used

Google Cloud Run
Google Cloud SQL (Postgres)
Setup and Configuration

Google Cloud SQL (Postgres) Setup
Create a PostgreSQL instance in Google Cloud SQL.
Set up the database schema and any initial data if required.
Google Cloud Run Setup
Enable Google Cloud Run API in your Google Cloud project.
Set up required IAM permissions for deploying and managing Cloud Run services.

Environment Variables
Ensure these environment variables are set either in your environment or in the Google Cloud Run service:

DB_HOST: The hostname of the Google Cloud SQL instance.
DB_PORT: Port, usually 5432 for PostgreSQL.
DB_USER: Database username.
DB_PASSWORD: Database password.
DB_NAME: Name of the database to connect to.
