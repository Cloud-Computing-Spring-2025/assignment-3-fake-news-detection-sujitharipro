from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat_ws
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Start Spark session
spark = SparkSession.builder.appName("FakeNews-Preprocessing").getOrCreate()

# Read dataset
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Ensure label column is cast to double
df = df.withColumn("label", col("label").cast("double"))

# TEMP FIX: Fill missing labels with 0.0 (fake) for testing purposes
df = df.fillna({'label': 0.0})

# Combine title and text into one column safely
df = df.withColumn("text_combined", concat_ws(" ", col("title").cast("string"), col("text").cast("string")))

# Tokenize
tokenizer = Tokenizer(inputCol="text_combined", outputCol="words_token")
df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words_token", outputCol="filtered_words")
df = remover.transform(df)

# Join tokens back into string
def join_words(words):
    return " ".join(words)

join_udf = udf(join_words, StringType())
df = df.withColumn("filtered_words_str", join_udf(col("filtered_words")))

# Select relevant columns
final_df = df.select("id", "filtered_words_str", "label")

# Drop any rows with null values in important columns (just in case)
final_df = final_df.dropna(subset=["filtered_words_str", "label"])

# Split data
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

# Optional: Print counts to verify
print(f"✅ Train data count: {train_data.count()}")
print(f"✅ Test data count: {test_data.count()}")

# Save datasets
train_data.coalesce(1).write.mode("overwrite").option("header", True).csv("train_data.csv")
test_data.coalesce(1).write.mode("overwrite").option("header", True).csv("test_data.csv")

print("✅ Preprocessing complete. Train and test datasets saved.")
