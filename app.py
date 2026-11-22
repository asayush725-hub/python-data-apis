from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 1. Fetch REST API data & load into Pandas DataFrame
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 15.8497,
        "longitude": 74.4977,
        "hourly": "temperature_2m,relative_humidity_2m",
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    df = pd.DataFrame(payload["hourly"])

    # 2. Clean & manipulate data with NumPy and Pandas
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    df["relative_humidity_2m"] = pd.to_numeric(df["relative_humidity_2m"], errors="coerce")

    df = df.drop_duplicates(subset=["time"]).dropna(subset=["time", "temperature_2m"])
    df["relative_humidity_2m"] = df["relative_humidity_2m"].fillna(
        df["relative_humidity_2m"].median()
    )

    df["date"] = df["time"].dt.date
    df["temp_category"] = np.where(
        df["temperature_2m"] >= 30,
        "Warm (>=30°C)",
        np.where(df["temperature_2m"] >= 20, "Moderate (20-29°C)", "Cool (<20°C)"),
    )

    daily_summary = (
        df.groupby("date", as_index=False)
        .agg(
            mean_temp_c=("temperature_2m", "mean"),
            max_temp_c=("temperature_2m", "max"),
            min_temp_c=("temperature_2m", "min"),
            mean_humidity_pct=("relative_humidity_2m", "mean"),
        )
        .round(2)
    )

    # 3. Visualize data with Matplotlib and Seaborn -> outputs/analysis_chart.png
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sns.lineplot(data=df, x="time", y="temperature_2m", ax=axes[0], color="#2563eb")
    axes[0].set_title("Hourly Temperature (°C)")
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(data=daily_summary, x="date", y="mean_temp_c", ax=axes[1], color="#0d9488")
    axes[1].set_title("Daily Mean Temperature (°C)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Mean Temperature (°C)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_dir / "analysis_chart.png", dpi=150)
    plt.show()

    # 4. Save CSVs & Markdown report inside outputs/
    df.to_csv(output_dir / "cleaned_dataset.csv", index=False)
    daily_summary.to_csv(output_dir / "daily_summary.csv", index=False)

    report_text = f"""# Python Data & APIs Analysis Report

- **Total Clean Records:** {len(df)}
- **Mean Temperature:** {np.mean(df["temperature_2m"]):.2f} °C
- **Peak Temperature:** {np.max(df["temperature_2m"]):.2f} °C
- **Lowest Temperature:** {np.min(df["temperature_2m"]):.2f} °C

## Daily Summary
{daily_summary.to_markdown(index=False)}
"""

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)


if __name__ == "__main__":
    main()