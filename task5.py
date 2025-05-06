from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNews-Detection").getOrCreate()

# Load preprocessed data
df = spark.read.csv("train_data.csv", header=True, inferSchema=True)

# Re-tokenize the preprocessed string (filtered_words_str)
tokenizer = Tokenizer(inputCol="filtered_words_str", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Index the label column for classification
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

# Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="label_index")

# Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, label_indexer, lr])

# Train model
model = pipeline.fit(df)

# Load test data
test_df = spark.read.csv("test_data.csv", header=True, inferSchema=True)

# Apply same pipeline to test data
predictions = model.transform(test_df)

# Evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"✅ Model accuracy: {accuracy:.4f}")

# Optionally: Show confusion matrix, F1 score, etc.
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction",
    metricName="f1"
)
f1_score = f1_evaluator.evaluate(predictions)
print(f"✅ Model F1 score: {f1_score:.4f}")
