import pandas as pd
import torch
import emoji
from collections import Counter
from datetime import timedelta
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
from src.core.config import settings
from monthly_analytics_engine import MonthlyAnalyticsEngine
from llm_wrapper import ollama_llm


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
        self.llm = ollama_llm


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
            "shared_core_lexicon": self.get_shared_core_lexicon(),
            "personal_lexicons": self.get_personal_lexicons(),
            "emotional_texture": self.get_emotional_texture(),
            "question_types": self.get_question_types(),
            "personality_profiles": self.get_personality_profiles(),
            "monthly_analytics": MonthlyAnalyticsEngine(self.df).to_dict()
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
    
    def get_shared_core_lexicon(self, top_k=50):
        logger.info("Computing shared core lexicon")

        df = self.df[~self.df["is_media"]]

        texts = (
            df.groupby("sender")["content"]
            .apply(lambda x: " ".join(x))
            .to_dict()
        )   

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=2
        )

        X = vectorizer.fit_transform(texts.values())
        terms = np.array(vectorizer.get_feature_names_out())

        sender_terms = {}
        for i, sender in enumerate(texts.keys()):
            scores = X[i].toarray().ravel()
            top = terms[np.argsort(scores)[-top_k:]]
            sender_terms[sender] = set(top)

        shared = set.intersection(*sender_terms.values())
        return list(shared)

    def get_personal_lexicons(self, top_k=50):
        df = self.df[~self.df["is_media"]].copy()
        senders = df["sender"].unique()

        texts = (
            df.groupby("sender")["content"]
            .apply(lambda x: " ".join(x.astype(str)))
            .to_list()
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1
        )

        X = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())

        sender_terms = {}
        for i, sender in enumerate(senders):
            scores = X[i].toarray().ravel()
            top_terms = terms[np.argsort(scores)[-top_k:]]
            sender_terms[sender] = set(top_terms)

        personal = {}
        for sender in senders:
            others = set.union(
                *[sender_terms[s] for s in senders if s != sender]
            )
            personal[sender] = list(sender_terms[sender] - others)

        return personal


    
   

    
    def get_emotional_texture(self):
        logger.info("Computing emotional texture")

        soften = [
    "maybe", "perhaps", "probably", "possibly",
    "i think", "i guess", "i feel",
    "idk", "not sure", "kinda", "sort of",
    "hmm", "uh", "um",
    "if you want", "up to you", "no pressure",
    "just checking", "just wondering"
         ]
        care = [
    "take care", "are you okay", "you okay",
    "did you eat", "ate?", "had food",
    "reach home", "reached?", "reach safely",
    "sleep well", "slept?", "rest",
    "stay safe", "be safe",
    "let me know", "tell me when",
    "don’t worry", "it's okay",
    "hope you're", "hope you are"
    ]


        stats = {}

        for sender in self.df["sender"].unique():
            text = " ".join(
                self.df[self.df["sender"] == sender]["content"]
                .astype(str)
                .str.lower()
            )

            stats[sender] = {
                "softeners": sum(text.count(w) for w in soften),
                "care_markers": sum(text.count(w) for w in care),
                "exclamations": text.count("!"),
                "ellipses": text.count("...")
            }

        return stats

    def select_high_signal_messages(
       self, sender: str,
        max_msgs: int = 20,
        min_chars: int = 80
    ):
        sdf = self.df[
            (self.df["sender"] == sender) &
            (~self.df["is_media"]) &
            (self.df["content"].str.len() >= min_chars)
        ].copy()

        if sdf.empty:
            return []

        # Heuristic scoring
        sdf["score"] = (
            sdf["content"].str.len() * 0.6 +
            sdf["content"].str.count("!") * 10 +
            sdf["content"].str.count("😂|😅|🥲|😎|😏|✨|💫|🤩") * 15
        )

        # Spread across time
        sdf = sdf.sort_values("timestamp")
        sampled = sdf.sample(
            n=min(max_msgs, len(sdf)),
            weights=sdf["score"],
            random_state=42
        )

        return sampled["content"].tolist()
    def analyze_sender_personality(self, sender: str, max_msgs: int = 20, min_chars: int = 60):
        """
        Analyze a sender's personality using LLM.
        Sends only the top high-signal messages in a single batch to avoid multiple calls.
        """

        messages = self.select_high_signal_messages(sender, max_msgs=max_msgs, min_chars=min_chars)
        if not messages:
            return {}

        interaction_roles = [
            "planner", "emotional anchor", "humor carrier",
            "motivator", "mediator", "advisor", "listener",
            "storyteller", "connector", "critic"
        ]

        numbered_messages = "\n".join(f"{i+1}. {m}" for i, m in enumerate(messages))

        prompt = f"""
    You are analyzing a person's conversational personality.

    Instructions:
    - Base analysis ONLY on provided messages
    - Identify stable patterns, not one-off moods
    - No moral judgment or speculation
    - Be precise, grounded, observational

    Messages:
    {numbered_messages}

    Output STRICT JSON:
    {{
    "dominant_tones": {{"playful":0-1, "caring":0-1, "logistical":0-1, "reflective":0-1, "detached":0-1}},
    "core_personality_traits": [8-12 concise traits],
    "communication_style": "5-7 sentence description",
    "interaction_role": {interaction_roles},
    "emotional_expression": "5 sentence summary"
    }}
    """.strip()

        # Retry logic
        for attempt in range(2):
            response = self.llm(prompt)
            if response.strip():
                try:
                    # Safe JSON extraction
                    import re, json
                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    if match:
                        return json.loads(match.group())
                    else:
                        logger.warning(f"No JSON object found in LLM response for {sender}")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse failed for {sender} (attempt {attempt+1}): {e}")
            else:
                logger.warning(f"Empty response for {sender} (attempt {attempt+1})")

        # Fallback if LLM fails
        logger.error(f"LLM personality analysis failed for {sender}")
        return {
            "dominant_tones": {t: 0 for t in ["playful", "caring", "logistical", "reflective", "detached"]},
            "core_personality_traits": [],
            "communication_style": "",
            "emotional_expression": "",
            "interaction_role": interaction_roles
        }
    
    def get_personality_profiles(self):
        if self.llm is None:
            logger.warning("LLM not provided, skipping personality profiles")
            return {}

        results = {}
        for sender in self.df["sender"].unique():
            logger.info(f"Analyzing personality for {sender}")
            results[sender] = self.analyze_sender_personality(sender)

        return results

    def get_question_types(self):
        logger.info("Computing question types")

        categories = {
            "logistic": ["when", "where", "time", "reach"],
            "emotional": ["okay", "fine", "feel"],
            "curious": ["why", "how"],
            "invitation": ["want", "shall", "come"]
        }

        results = {}

        for sender in self.df["sender"].unique():
            qs = self.df[
                (self.df["sender"] == sender)
                & (self.df["content"].str.contains("?", regex=False))
            ]["content"].str.lower()

            counts = {k: 0 for k in categories}
            for q in qs:
                for cat, keys in categories.items():
                    if any(k in q for k in keys):
                        counts[cat] += 1

            results[sender] = counts

        return results
    

