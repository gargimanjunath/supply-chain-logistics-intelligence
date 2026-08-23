"""Load the cleaned dataset into a SQLite database (sql/ layer).
SQLite is used here because this sandbox has no persistent MySQL server;
the queries in sql/analysis_queries.sql are written in standard ANSI SQL
and are portable to MySQL 8+ with no changes (window functions supported
in both).
"""
import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/cleaned_supply_chain.csv")

# Slim column names for SQL friendliness
df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
              for c in df.columns]

conn = sqlite3.connect("data/processed/supply_chain.db")
df.to_sql("orders_fact", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON orders_fact(Order_Id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_order_month ON orders_fact(order_month)")
conn.commit()
print("Rows loaded:", conn.execute("SELECT COUNT(*) FROM orders_fact").fetchone()[0])
print("Columns:", [r[1] for r in conn.execute("PRAGMA table_info(orders_fact)").fetchall()])
conn.close()
