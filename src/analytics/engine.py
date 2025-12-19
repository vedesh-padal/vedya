import pandas as pd
import torch
import emoji
from collections import Counter
from datetime import timedelta
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
from src.core.config import settings


class AnalyticsEngine:
    def __init__(self):
        self.raw_path = settings.PROCESSED_DATA_DIR / "chat_history_raw.parquet"
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at: {self.raw_path}")

        self.df = pd.read_parquet(self.raw_path)
        # Ensure timestamp is datetime
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Pre-calculate common columns for speed
        # 'diff' calculates the time difference between current row and previous row
        self.df["time_diff"] = self.df["timestamp"].diff()
        # 'sender_changed' is True if the sender is different from the previous message
        self.df["sender_changed"] = self.df["sender"] != self.df["sender"].shift(1)

    def run_analysis(self, sentiment_sample=500):
        """
        Orchestrates the full analysis pipeline.
        """
        logger.info("--- Generating Deep Relationship Insights ---")

        profile = {
            "volume_metrics": self._get_volume_metrics(),
            "temporal_metrics": self._get_temporal_metrics(),
            "behavioral_metrics": self._get_behavioral_metrics(),
            "linguistic_metrics": self._get_linguistic_metrics(),
            "sentiment_metrics": self._get_sentiment_metrics(sentiment_sample),
        }
        return profile

    def _get_volume_metrics(self):
        """Metric 1-3: Basic Counts & Ratios"""
        counts = self.df["sender"].value_counts().to_dict()
        total = len(self.df)

        # Calculate Media Ratio (assuming <Media omitted> tag)
        media_msgs = self.df[self.df['content'].str.contains("<Media omitted>", na=False)]
        media_counts = media_msgs["sender"].value_counts().to_dict()

        return {
            "total_messages": total,
            "message_distribution": counts,
            "media_messages": media_counts,
            "top_sender": max(
                counts, key=lambda x: counts[x]
            ),  # a shorter syntax here could have been: max(counts, key=counts.get) - but Pylance is complaining
        }

    def _get_temporal_metrics(self):
        """ "Metric 4-6: Chronotypes & Peak Times"""
        # Hourly Heatmap
        hourly = self.df["timestamp"].dt.hour.value_counts().sort_index().to_dict()  # type: ignore
        # Peak Hour: The hour with the most messages
        peak_hour = max(
            hourly, key=hourly.get
        )  # a custom key function to dictate what max() should compare.

        # Chronotype Logic
        if 5 <= peak_hour < 12:
            chronotype = "Morning Birds"
        elif 12 <= peak_hour < 17:
            chronotype = "Afternoon Talkers"
        elif 17 <= peak_hour < 21:
            chronotype = "Evening Chillers"
        else:
            chronotype = "Night Owls"

        # Busy Days: Count messages per day of the week``
        daily = self.df["timestamp"].dt.day_name().value_counts().to_dict()  # type: ignore

        start_date = (
            self.df["timestamp"].min().strftime("%Y-%m-%d")
        )  # First message date

        return {
            "chronotype": chronotype,
            "peak_hour": peak_hour,
            "busiest_day": max(daily, key=daily.get),
            "hourly_activity": hourly,
            "start_date": start_date,
        }

    def _get_behavioral_metrics(self):
        """Metric 7-10: Latency, Initiation, Double Texting"""
        df = self.df.copy()

        # 1. Initiation Rate (who starts convo after > 8 hours silence?)
        # Logic: If time_diff > 8 hours, the current sender 'Initiated'
        initiations = df[df["time_diff"] > timedelta(hours=8)]
        init_counts = initiations["sender"].value_counts().to_dict()

        # 2. Response Latency (How fast do they reply?)
        # Logic: Only look at rows where sender CHANGED and time_diff > 7 hours (ignore sleep)
        replies = df[df["sender_changed"] & (df["time_diff"] < timedelta(hours=7))]
        # Convert to minutes
        avg_latency = (
            replies.groupby("sender")["time_diff"]
            .apply(lambda x: x.dt.total_seconds().mean() / 60)
            .to_dict()
        )

        # 3. Double Texting (Sending multiple msgs in a row)
        # Logic: Sender did NOT change, and time diff is small (< 5 mins)
        double_texts = df[
            ~df["sender_changed"] & (df["time_diff"] < timedelta(minutes=5))
        ]
        double_text_counts = double_texts["sender"].value_counts().to_dict()

        return {
            "conversation_initiations": init_counts,
            "avg_reply_time_minutes": {k: round(v, 1) for k, v in avg_latency.items()},
            "double_text_count": double_text_counts,
        }

    def _get_linguistic_metrics(self):
        """Metric 11-14: Emojis, Vocab, Questions"""

        def extract_emojis(text):
            return [c["emoji"] for c in emoji.emoji_list(text)]

        stats = {}
        senders = self.df["sender"].unique()

        for person in tqdm(senders, desc="Analyzing Linguistics"):
            person_df = self.df[self.df["sender"] == person]
            all_text = " ".join(person_df["content"].astype(str))

            # 1. Top Emojis
            all_emojis = extract_emojis(all_text)
            top_emojis = Counter(all_emojis).most_common(5)

            # 2. Question Ratio
            question_count = person_df["content"].str.count(r"\?").sum()

            # 3. Avg Message Length (in number of words)
            text_df = person_df[~person_df["is_media"]]  #
            avg_len = text_df["content"].str.split().str.len().mean()

            stats[person] = {
                "top_emojis": [e[0] for e in top_emojis],
                "questions_asked": int(question_count),
                "avg_words_per_msg": round(avg_len, 1),
            }

        return stats

    def _get_sentiment_metrics(self, sample_size):
        """Metric 15: The Vibe Check (AI Model)"""
        logger.info("Loading Sentiment Model...")
        device = 0 if torch.cuda.is_available() else -1

        MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        pipe = pipeline(
            "sentiment-analysis",  # type: ignore
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,  # Return all scores
        )

        # Filter short messages
        long_msgs = self.df[self.df["content"].str.len() > 15]

        if sample_size and len(long_msgs) > sample_size:
            target_df = long_msgs.sample(sample_size, random_state=42)
        else:
            target_df = long_msgs

        logger.info(f"Running sentiment on {len(target_df)} messages...")
        
        texts = target_df["content"].tolist()
        
        # results = pipe(texts, batch_size=16)  # Batch size speeds it up

        # using TQDM for progress bar
        batch_size = 16
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
            batch = texts[i : i + batch_size]
            batch_results = pipe(batch, truncation=True, max_length=512)
            results.extend(batch_results)

        # Aggregate
        score_sums = {"positive": 0, "negative": 0, "neutral": 0}
        for res in results:
            top = max(res, key=lambda x: x["score"])
            score_sums[top["label"]] += 1

        total = len(texts)
        return {k: round((v / total) * 100, 1) for k, v in score_sums.items()}
