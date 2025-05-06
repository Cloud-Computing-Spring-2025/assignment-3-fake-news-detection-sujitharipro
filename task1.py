from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("FakeNewsDetection").getOrCreate()

# Load CSV with inferred schema
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create temporary view
df.createOrReplaceTempView("news_data")

# Basic queries
df.show(5)
print("Total articles:", df.count())
spark.sql("SELECT DISTINCT label FROM news_data").show()

# Save output
df.limit(5).write.csv("task1_output.csv", header=True, mode="overwrite")
