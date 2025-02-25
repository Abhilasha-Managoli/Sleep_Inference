import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from datetime import datetime

# ---------------------------
# Step 1: Load Chrome and YouTube History
# ---------------------------

# Load Chrome history (Filtered Jan 23 - Feb 24)
chrome_file = "Chrome_History_Jan23_Feb24.csv"
df_chrome = pd.read_csv(chrome_file)


# Function to Load YouTube Watch History from HTML
def load_youtube_history_from_html(filename):
    """
    Parse YouTube watch history from an HTML file and extract timestamps.
    """
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
    df_youtube = df_youtube.dropna(subset=["visit_date"])  # Remove missing timestamps
    return df_youtube


# Load YouTube history from the correct HTML file
youtube_file = "watch-history.html"
df_youtube = load_youtube_history_from_html(youtube_file)

# Save YouTube data as CSV for future use
df_youtube.to_csv("YouTube_History.csv", index=False)

# Convert timestamps to datetime
df_chrome["visit_date"] = pd.to_datetime(df_chrome["visit_date"])
df_youtube["visit_date"] = pd.to_datetime(df_youtube["visit_date"])

# Add source columns
df_chrome["source"] = "Chrome"
df_youtube["source"] = "YouTube"

# Combine datasets
df = pd.concat([df_chrome, df_youtube], ignore_index=True)
df.sort_values(by="visit_date", inplace=True)

# ---------------------------
# Step 2: Aggregate Activity by Hour
# ---------------------------

# Extract date and hour from timestamps
df["date"] = df["visit_date"].dt.date
df["hour"] = df["visit_date"].dt.hour

# Count activity per hour per day
activity_by_hour = df.groupby(["date", "hour"]).size().reset_index(name="activity_count")


# ---------------------------
# Step 3: Infer Sleep Periods
# ---------------------------

def infer_sleep_periods(activity_df, threshold=1):
    """
    Infers sleep periods by finding the longest continuous block of hours with
    low activity (activity_count <= threshold) for each day.
    """
    sleep_data = []

    for date, group in activity_df.groupby("date"):
        low_activity = [False] * 24
        for hour in range(24):
            count_series = group[group["hour"] == hour]["activity_count"]
            count = count_series.iloc[0] if not count_series.empty else 0
            low_activity[hour] = (count <= threshold)

        # Find longest block of low activity (likely sleep)
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


# Infer sleep periods
sleep_df = infer_sleep_periods(activity_by_hour, threshold=1)


# ---------------------------
# Step 4: Compare to Ground Truth (if available)
# ---------------------------

# If you kept a manual sleep log or used a smartwatch, you can compare actual sleep times.
# Example structure for ground truth data:
# ground_truth = pd.read_csv("sleep_log.csv")  # This should contain 'date', 'actual_sleep_start', 'actual_sleep_end'
# merged_df = sleep_df.merge(ground_truth, on="date", how="left")
# merged_df["error_hours"] = abs(merged_df["sleep_start"] - merged_df["actual_sleep_start"])
# print(merged_df)

# ---------------------------
# Step 5: Visualize Activity Heatmap
# ---------------------------

def create_heatmap(activity_df):
    """
    Creates and displays a heatmap of Google activity by hour.
    """
    pivot = activity_df.pivot(index="date", columns="hour", values="activity_count").fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlGnBu")
    plt.title("Google/Youtube Activity from Jan 23, 2025 - Feb 23, 2025 Heatmap")
    plt.xlabel("Hour of Day")
    plt.ylabel("Date")
    plt.show()


# Generate heatmap
create_heatmap(activity_by_hour)

# ---------------------------
# Step 6: Save Inferred Sleep Data
# ---------------------------
sleep_output_file = "Inferred_Sleep_Data.csv"
sleep_df.to_csv(sleep_output_file, index=False)

print(f"Sleep data saved to {sleep_output_file}")
