import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///db/agricole.db")


def show_table(table_name):
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    print(f"\n===== {table_name} =====")
    print(df)


if __name__ == "__main__":
    show_table("analyses_culture")
    show_table("analyses_vache")
    show_table("analyses_feuille")
