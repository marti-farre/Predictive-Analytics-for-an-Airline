from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
from pyspark.sql import functions as F
import glob
import time


def read_plane_sensors (spark, URI_PATH_TO_DIRECTORY, num_partitions):
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

    #Join all csv indormation
    df_sensors = spark.createDataFrame([], df_structure)
    for file in csv_files:
        df = spark.read.csv(file, header=True, sep=";", schema = csv_structure)
        df = df.select("date", "value").withColumnRenamed("date", "timeid")
        df = df.withColumn("aircraftid", F.lit(file[-10:-4]))
        df_sensors = df_sensors.union(df)

    #Group by aircraft and timeid
    df_sensors = df_sensors.coalesce(num_partitions).groupBy("aircraftid","timeid").agg(F.avg("value"). alias("avg_sensor_val"))

    return df_sensors.sort(['aircraftid', 'timeid']) #We sort the information to boost performance in future joins



def obtain_DW_data (spark):
    print("Setting the obtention of data from the DW...")
    try:
        # Load the data from the PostgreSQL table aircraftutilization
        df_aircraftutilization = spark.read.format("jdbc")\
            .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")\
            .option("dbtable", "aircraftutilization") \
            .option("user", "armand.de.asis") \
            .option("password", "DB250202") \
            .option("driver", "org.postgresql.Driver") \
            .load()
    except:
        print("Error connecting to the database, check your VPN or server status")

    #We select and transform the data
    df_kpi = df_aircraftutilization.select(['aircraftid', 'timeid', 'flighthours', 'flightcycles', 'delays'])
    df_kpi = df_kpi.withColumn("flightcycles", F.col("flightcycles").cast("integer")).\
        withColumn("delays", F.col("delays").cast("integer"))

    return df_kpi.sort(['aircraftid', 'timeid']) #We sort the information to boost performance in future joins

def obtain_OP_AMOS_data (spark):
    print("Setting the obtention of data from the Operation Interruption AMOS table...")
    try:
        # Load the data from the PostgreSQL table operation interruption
        df_op_amos = spark.read.format("jdbc")\
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")\
                .option("dbtable", "oldinstance.operationinterruption") \
                .option("user", "armand.de.asis") \
                .option("password", "DB250202") \
                .option("driver", "org.postgresql.Driver") \
                .load()
    except:
        print("Error connecting to the database, check your VPN or server status")

    #We filter, select and transform the data to be able to do the labelling
    df_maintenances = df_op_amos.select(['aircraftregistration','subsystem', 'starttime','kind'])\
                        .filter(F.col("subsystem") == "3453").drop('subsystem')\
                        .filter(F.col("kind").isin(["Safety", "Delay", "AircraftOnGround"]))\
                        .withColumn("timeid", F.to_date(df_op_amos.starttime)).drop("starttime")\
                        .withColumnRenamed("aircraftregistration", "aircraftid")
    return df_maintenances.sort(['aircraftid', 'timeid']) #We sort the information to boost performance in future joins


def data_labelling(df_features, df_maintenances):
    print("Setting the labelling of the data...")
    # Create a reference dataframe with the previous seven days of each Maintenance Event of an specific airplane
    df_7_maintenances = df_maintenances
    df_7_maintenances.cache()
    # Add rows for the previous seven days from the day thta has ocurred a maintenance event
    for i in range(1, 8):
        df_prev = df_maintenances.withColumn("timeid", F.date_add(df_maintenances["timeid"], -i))
        df_7_maintenances = df_7_maintenances.union(df_prev)

    # Drop duplicates if they have various kinds of Maintenances for same airplane and same day
    df_7_maintenances = df_7_maintenances.dropDuplicates(subset=["aircraftid", "timeid"])

    # Join the flights with all features and the actual and past 7 maintenances.    
    df_training = df_7_maintenances.join(df_features, on=["aircraftid", "timeid"], how = "right")
    df_7_maintenances.unpersist()

    #Set the label of Maintenance or nonMaintenance
    df_training = df_training.withColumn("label", F.when(F.col("kind").isNotNull(), "Maintenance").otherwise("NonMaintenance"))
    return df_training.drop("aircraftid", "timeid", "kind")




def run(spark, URI_PATH_TO_DIRECTORY, num_partitions):
    start = time.time() 

    #Read the csv, DW and AMOS info
    df_sensors = read_plane_sensors (spark, URI_PATH_TO_DIRECTORY, num_partitions)
    df_kpi = obtain_DW_data(spark)
    df_maintenances = obtain_OP_AMOS_data(spark)

    #Join csv and DW info
    print("Setting the join of KPIs and sensor data to generate the features...")
    df_features = df_kpi.join(df_sensors, on=["aircraftid", "timeid"], how= "inner")

    #Label the data
    df_training = data_labelling(df_features, df_maintenances)

    #Save the data
    df_training.coalesce(1).write.csv("data_training", mode="overwrite", header=True) # We store all data into a one csv
    

    end = time.time()
    print("The execution time is:", end - start)

    