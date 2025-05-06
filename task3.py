from pyspark.sql import SparkSession
from pyspark.sql.functions import split, concat_ws
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.functions import vector_to_array

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNews-Task3").getOrCreate()

# Load the cleaned tokenized data
df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

# Convert filtered_words string back to array
df = df.withColumn("filtered_words", split(df["filtered_words_str"], ","))

# TF-IDF transformation
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
tf_df = hashingTF.transform(df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# Convert vector to array of numbers (to make it CSV-writable)
tfidf_df = tfidf_df.withColumn("features_array", vector_to_array("features"))

# Convert label to numeric
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(tfidf_df).transform(tfidf_df)

# Join the array into a string so it's writable to CSV
output_df = indexed.withColumn("features_str", concat_ws(",", "features_array"))

# Write selected columns to CSV
output_df.select("id", "filtered_words_str", "features_str", "label_index") \
         .write.csv("task3_output.csv", header=True, mode="overwrite")

spark.stop()
