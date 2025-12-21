import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from loguru import logger

class MonthlyAnalyticsEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # 🔑 make month JSON-safe at source
        self.df["month"] = (
            self.df["timestamp"]
            .dt.to_period("M")
            .astype(str)
        )


    def message_volume(self):
        print("Computing monthly message volume")

        return (
            self.df
            .groupby(["month", "sender"])
            .size()
            .unstack(fill_value=0)
        )

    def avg_message_length(self):
        print("Computing average message length")

        self.df["length"] = self.df["content"].str.split().str.len()

        return (
            self.df
            .groupby(["month", "sender"])["length"]
            .mean()
            .round(1)
            .unstack()
        )

    def monthly_vocab_shift(self, top_k=20):
        result = {}

        for month, mdf in self.df.groupby("month"):
            texts = mdf["content"].dropna().astype(str)

            # Skip tiny or useless months
            if len(texts) < 5:
                result[str(month)] = []
                continue

            vectorizer = TfidfVectorizer(
                stop_words="english",
                min_df=1
            )

            try:
                tfidf = vectorizer.fit_transform(texts)
            except ValueError:
                # No usable vocabulary this month
                result[str(month)] = []
                continue

            scores = tfidf.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()

            if len(terms) == 0:
                result[str(month)] = []
                continue

            top = sorted(
                zip(terms, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            result[str(month)] = [w for w, _ in top]

        return result



    def silence_gaps(self, gap_hours: int = 6) -> dict:
        logger.info("Computing silence gaps")

        df = self.df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        df["gap_hours"] = df["timestamp"].diff().dt.total_seconds() / 3600

        silence_df = df.loc[df["gap_hours"] > gap_hours].copy()

        if silence_df.empty:
            logger.warning("No silence gaps detected")
            return {}

        silence_df.loc[:, "month"] = (
            silence_df["timestamp"]
            .dt.to_period("M")
            .astype(str)         
        )

        counts = silence_df.groupby("month").size()

        result = {month: int(count) for month, count in counts.items()}

        logger.info(f"Silence gaps computed for {len(result)} months")
        return result

