from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from pyspark.sql import functions as F
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.feature import VectorAssembler
import glob
import time

def read_plane_sensors (spark, URI_PATH_TO_DIRECTORY, aircraftid, date):
    print("Setting the obtention of airplane sensors from data files...")
    csv_structure = StructType([
            StructField("date", DateType(), True),
            StructField("series", StringType(), True),
            StructField("value", DoubleType(), True),
    ])
    df_structure = StructType([
            StructField("timeid", DateType(), True),
            StructField("value", DoubleType(), True),
            StructField("aircraftid", StringType(), True),
    ])

    #Obtain all csv
    csv_files = glob.glob(URI_PATH_TO_DIRECTORY[11:]+"/resources/trainingData/*csv")

    assert len(csv_files) != 0, "There is no .csv sensor data to analyse in the mentioned directory"

    df_flight_date_sensor = spark.createDataFrame([], df_structure)
    for file in csv_files:
        if file[-10:-4] == aircraftid:  # only select the file corresponding to the specified aircraftid
            df = spark.read.csv(file, header=True, sep=";", schema = csv_structure)
            df = df.select("date", "value").withColumnRenamed("date", "timeid")
            df = df.withColumn("aircraftid", F.lit(file[-10:-4]))
            df_flight_date_sensor = df_flight_date_sensor.union(df)

    #Filter by date
    df_flight_date_sensor = df_flight_date_sensor.where(df_flight_date_sensor.timeid == date)  # only select rows with the specified date
    

    assert len(df_flight_date_sensor.head(1)) != 0, "This plane doesn't have data on that date!"
    # Give the average
    df_flight_date_sensor = df_flight_date_sensor.groupBy("aircraftid","timeid").agg(F.avg("value"). alias("avg_sensor_val"))

    return df_flight_date_sensor


def obtain_DW_data (spark, aircraftid, date):
    print("Setting the obtention of data from the DW...")
    try:
        # Load the data from the PostgreSQL table aircraftutilization
        df_aircraftutilization = spark.read.format("jdbc")\
            .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")\
            .option("dbtable", "aircraftutilization") \
            .option("user", "USEREXAMPLE") \
            .option("password", "passwordexemple") \
            .option("driver", "org.postgresql.Driver") \
            .load()
    except:
        print("Error connecting to the database, check your VPN or server status")

    #Filter by aircraftid and date
    df_kpi = df_aircraftutilization.where((df_aircraftutilization.aircraftid == aircraftid) & (df_aircraftutilization.timeid == date))

    assert len(df_kpi.head(1)) != 0, "This plane doesn't have data on that date!"

    #We select and transform the data
    df_kpi = df_kpi.select(['aircraftid', 'timeid', 'flighthours', 'flightcycles', 'delays'])
    df_kpi = df_kpi.withColumn("flightcycles", F.col("flightcycles").cast("integer")).\
        withColumn("delays", F.col("delays").cast("integer"))

    return df_kpi



def get_input_features(spark, URI_PATH_TO_DIRECTORY):

    aircraftid = input("Enter an aircraft ID (with format XX-XXX): ")
    date = input("Enter a date (with format YYYY-MM-DD): ")

    start = time.time()

    #Read the csv and DW
    df_sensors = read_plane_sensors(spark, URI_PATH_TO_DIRECTORY, aircraftid, date)
    df_kpi = obtain_DW_data(spark, aircraftid, date)

    print("Setting the join of KPIs and sensor data to generate the features...")
    df_features = df_kpi.join(df_sensors, on=["aircraftid", "timeid"], how= "inner")
    return df_features, start

def data_preparation(df_features):
    print("Creating the features df for prediction...")
    assembler = VectorAssembler(
    inputCols=["flighthours","flightcycles","delays","avg_sensor_val"],
    outputCol="features")
    df_predict = assembler.transform(df_features)
    df_predict = df_predict.select(['features'])
    return df_predict

def evaluate_case(model, df_predict):
    # Predict the class for the instance
    print("Predicting if we will have a maintenance...")
    prediction = model.transform(df_predict)
    return prediction.select("prediction").first()[0]

def run(spark, URI_PATH_TO_DIRECTORY):
    
    df_features, start = get_input_features(spark, URI_PATH_TO_DIRECTORY)
    # Load the model
    model = DecisionTreeClassificationModel.load(URI_PATH_TO_DIRECTORY+"/dtc")
    df_predict = data_preparation(df_features)
    maintenance = evaluate_case(model, df_predict)
    print("-> The aircraft will fly with no maintenance needed in next seven days!") if maintenance == 0 else print("-> The aircraft will require maintenace in next seven days!")

    end = time.time()
    print("The execution time is:", end - start)
