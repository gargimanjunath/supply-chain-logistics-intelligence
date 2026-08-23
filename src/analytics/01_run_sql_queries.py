"""Execute every query in sql/analysis_queries.sql against the SQLite
analytical DB and save results to reports/sql_results.json (used both
for the final PDF report and to sanity-check the SQL layer)."""
import sqlite3
import pandas as pd
import re
import json

conn = sqlite3.connect("data/processed/supply_chain.db")

with open("sql/business_queries.sql") as f:
    content = f.read()

# Split on ';' at end of statements, keep comments as query titles
blocks = [b.strip() for b in content.split(";") if b.strip()]

results = {}
current_title = None
for block in blocks:
    lines = block.split("\n")
    title_lines = [l.strip("- ").strip() for l in lines
                   if l.strip().startswith("--") and l.strip().strip("-").strip()
                   and "====" not in l]
    sql_lines = [l for l in lines if not l.strip().startswith("--")]
    sql = "\n".join(sql_lines).strip()
    if not sql:
        continue
    title = title_lines[0] if title_lines else sql[:40]
    try:
        df = pd.read_sql_query(sql, conn)
        results[title] = df.head(15).to_dict(orient="records")
        print(f"OK   | {title}  -> {len(df)} rows")
    except Exception as e:
        print(f"FAIL | {title}  -> {e}")
        results[title] = f"ERROR: {e}"

with open("reports/sql_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

conn.close()
print("\nSaved to reports/sql_results.json")
