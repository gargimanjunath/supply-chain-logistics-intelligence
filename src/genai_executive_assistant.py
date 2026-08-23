"""
Optional Phase - GenAI Executive Assistant.

Converts VALIDATED analytical outputs (KPIs, stats tests, forecast, ML
results already computed and saved to /reports) into a plain-language
executive summary via an LLM call. The model is explicitly instructed to
use only the numbers it is given -- it must not invent figures.

Requires ANTHROPIC_API_KEY to be set in the environment to actually call the
API; running this script without a key will print the prompt it *would* send
so the logic can still be reviewed/tested offline.
"""
import json
import os

def load_validated_outputs():
    with open("reports/kpi_summary.json") as f:
        kpis = json.load(f)
    with open("reports/statistical_test_results.json") as f:
        stats_ = json.load(f)
    with open("reports/forecast_results.json") as f:
        forecast = json.load(f)
    with open("reports/ml_model_comparison.json") as f:
        ml = json.load(f)
    return kpis, stats_, forecast, ml


def build_prompt(kpis, stats_, forecast, ml):
    return f"""You are a supply-chain analytics assistant. Using ONLY the
validated data below, write a concise executive summary (300-400 words) for
a supply-chain leadership team. Do not invent any numbers not provided here.

VALIDATED KPIs:
{json.dumps(kpis, indent=2)}

STATISTICAL TEST RESULTS:
{json.dumps({k: v for k, v in stats_.items() if 'descriptive' not in k}, indent=2)}

DEMAND FORECAST:
{json.dumps(forecast, indent=2)}

ML LATE-DELIVERY MODEL RESULTS:
{json.dumps({k: v for k, v in ml.items() if k not in ('descriptive_statistics',)}, indent=2)}

Structure your summary as:
1. Business snapshot (revenue, orders, on-time performance)
2. Key drivers of late delivery (statistically validated)
3. Demand outlook (next 6 months)
4. ML model recommendation for proactive intervention
5. Two concrete operational recommendations
"""


def main():
    kpis, stats_, forecast, ml = load_validated_outputs()
    prompt = build_prompt(kpis, stats_, forecast, ml)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set - printing the prompt that would be sent:\n")
        print(prompt)
        return

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.content[0].text
    with open("reports/genai_executive_summary.txt", "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
