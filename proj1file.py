import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from datetime import datetime

chrome_file = "Chrome_History_Jan23_Feb24.csv"
df_chrome = pd.read_csv(chrome_file)


def load_youtube_history_from_html(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    entries = soup.find_all("div", class_="content-cell")  # May need to adjust this tag
    titles = []
    timestamps = []

    for entry in entries:
        title_tag = entry.find("a")  # Video title link
        title = title_tag.text.strip() if title_tag else "No Title"
        titles.append(title)

        time_text = entry.text.split("Watched")[-1].strip() if "Watched" in entry.text else None
        try:
            timestamp = datetime.strptime(time_text, "%b %d, %Y, %I:%M:%S %p") if time_text else None
        except ValueError:
            timestamp = None

        timestamps.append(timestamp)

    df_youtube = pd.DataFrame({"title": titles, "visit_date": timestamps})
    df_youtube = df_youtube.dropna(subset=["visit_date"])
    return df_youtube


youtube_file = "watch-history.html"
df_youtube = load_youtube_history_from_html(youtube_file)

df_youtube.to_csv("YouTube_History.csv", index=False)

df_chrome["visit_date"] = pd.to_datetime(df_chrome["visit_date"])
df_youtube["visit_date"] = pd.to_datetime(df_youtube["visit_date"])

df_chrome["source"] = "Chrome"
df_youtube["source"] = "YouTube"

df = pd.concat([df_chrome, df_youtube], ignore_index=True)
df.sort_values(by="visit_date", inplace=True)

df["date"] = df["visit_date"].dt.date
df["hour"] = df["visit_date"].dt.hour

activity_by_hour = df.groupby(["date", "hour"]).size().reset_index(name="activity_count")

def infer_sleep_periods(activity_df, threshold=1):
    sleep_data = []

    for date, group in activity_df.groupby("date"):
        low_activity = [False] * 24
        for hour in range(24):
            count_series = group[group["hour"] == hour]["activity_count"]
            count = count_series.iloc[0] if not count_series.empty else 0
            low_activity[hour] = (count <= threshold)

        max_length = 0
        current_length = 0
        current_start = None
        sleep_start = None
        sleep_end = None

        for hour in range(24):
            if low_activity[hour]:
                if current_start is None:
                    current_start = hour
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    sleep_start = current_start
                    sleep_end = hour
                current_start = None
                current_length = 0

        if current_length > max_length:
            max_length = current_length
            sleep_start = current_start
            sleep_end = (current_start + current_length) % 24

        sleep_data.append({
            "date": date,
            "sleep_start": sleep_start,
            "sleep_end": sleep_end,
            "duration_hours": max_length
        })

    return pd.DataFrame(sleep_data)


sleep_df = infer_sleep_periods(activity_by_hour, threshold=1)


def create_heatmap(activity_df):
    pivot = activity_df.pivot(index="date", columns="hour", values="activity_count").fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlGnBu")
    plt.title("Google/Youtube Activity from Jan 23, 2025 - Feb 23, 2025 Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.show()


create_heatmap(activity_by_hour)

sleep_output_file = "Inferred_Sleep_Data.csv"
sleep_df.to_csv(sleep_output_file, index=False)

print(f"Sleep data saved to {sleep_output_file}")
