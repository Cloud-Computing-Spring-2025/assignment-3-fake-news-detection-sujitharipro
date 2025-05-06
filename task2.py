from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Step 1: Start Spark session
spark = SparkSession.builder.appName("FakeNewsTask2").getOrCreate()

# Step 2: Load CSV file
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Step 3: Convert text to lowercase
df_cleaned = df.withColumn("text", lower(col("text")))

# Step 4: Tokenize text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(df_cleaned)

# Step 5: Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered = remover.transform(tokenized)

# Step 6: Convert filtered_words array to a comma-separated string
output_df = filtered.withColumn("filtered_words_str", concat_ws(",", col("filtered_words")))

# Step 7: Write only necessary columns to CSV
output_df.select("id", "title", "filtered_words_str", "label") \
    .write.csv("task2_output.csv", header=True, mode="overwrite")

# Step 8: Stop Spark session
spark.stop()
