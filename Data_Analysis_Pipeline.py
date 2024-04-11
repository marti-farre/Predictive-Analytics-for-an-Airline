from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark.sql.functions import when
import time

def data_preparation (spark, URI_PATH_TO_DIRECTORY):
    # Read the data from the csv files
    schema_data = StructType([
        StructField("flighthours", DoubleType(), True),
        StructField("flightcycles", IntegerType(), True),
        StructField("delays", IntegerType(), True),
        StructField("avg_sensor_val", DoubleType(), True),
        StructField("label", StringType(), True)
    ])
    df_analysis = spark.read.csv(URI_PATH_TO_DIRECTORY[11:]+"/data_training/*.csv", header=True, schema=schema_data)

    # Create the features vector
    assembler = VectorAssembler(
    inputCols=["flighthours","flightcycles","delays","avg_sensor_val"],
    outputCol="features")
    df_analysis = assembler.transform(df_analysis)

    # Create the label vector
    df_analysis = df_analysis.select(['features', 'label'])\
        .withColumn("num_label", when(df_analysis.label == 'Maintenance', 1).otherwise(0)).drop('label')

    return df_analysis.randomSplit([0.7, 0.3]) # Split the data into training and validation


def training (URI_PATH_TO_DIRECTORY, df_train):
    # Create the decision tree model
    dt = DecisionTreeClassifier(labelCol="num_label", featuresCol="features")

    # Train the model
    model = dt.fit(df_train)

    # Save the model
    model.write().overwrite().save(URI_PATH_TO_DIRECTORY[11:]+"/dtc")

    return model

def validation(df_val, model):
    # Make predictions on the test data
    predictions = model.transform(df_val)
    
    # Create a binary class classification evaluator for accuracy
    evaluator_accuracy = BinaryClassificationEvaluator(labelCol="num_label", metricName="areaUnderROC")

    # Create a binary class classification evaluator for recall
    evaluator_recall = BinaryClassificationEvaluator(labelCol="num_label", metricName="areaUnderPR")

    # Evaluate the model
    accuracy = evaluator_accuracy.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)

    # Print the results
    print("Accuracy:", accuracy)
    print("Recall:", recall)


def run(spark, URI_PATH_TO_DIRECTORY): 
    start = time.time()
    #Get, split and prepare data for the training with a decision tree
    df_train , df_val  = data_preparation(spark, URI_PATH_TO_DIRECTORY)

    #Train the model and store it
    model = training(URI_PATH_TO_DIRECTORY, df_train)

    #Validate the model and show the accuracy and recall values
    validation(df_val, model)
    end = time.time()
    print("The execution time is:", end - start)