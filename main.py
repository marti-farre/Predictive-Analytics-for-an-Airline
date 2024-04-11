import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import Data_Management_Pipeline
import Data_Analysis_Pipeline
import Run_time_Classifier_Pipeline

#Set the path to the directory where the project is located, in windows format and in URI
WIN_PATH_TO_DIRECTORY = 'C:\\Users\\arman\\Downloads\\Marti_farrer_Armand_de_asis'
URI_PATH_TO_DIRECTORY = 'file:///C://Users/arman/Downloads/Marti_farrer_Armand_de_asis'

HADOOP_HOME = WIN_PATH_TO_DIRECTORY+'\\resources\\hadoop_home'
JDBC_JAR = WIN_PATH_TO_DIRECTORY+"\\resources\\postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python"
PYSPARK_DRIVER_PYTHON = "python"

if(__name__== "__main__"):
    #Set environment variables
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    #Set number of cores in local machine
    num_cores = input("Choose the number of cores in your local machine:")
    assert num_cores.isdigit() or int(num_cores)> 1, "Error: the number of data partitions is not a number or is not valid"

    #We set the number of partitions equal to the number of cores as we have seen this value to be the best one in perfomance
    num_partitions = int(num_cores)

    #We let the user change it if he wants to
    print("We have set the number of partitions to:", num_partitions)
    accepted = input("Would you like to change it? (Y/N)")
    if accepted == "Y" or accepted == "y":
        num_partitions = input("Insert the number of partitions you want:")
        assert num_partitions.isdigit() or int(num_partitions)> 1, "Error: the number of data partitions is not a number or is not valid"
        num_partitions = int(num_partitions)

    #Set Spark configuration
    conf = SparkConf()  
    conf.set("spark.jars", JDBC_JAR)
    conf.set("spark.master", "local["+str(num_cores)+"]")
    conf.set("spark.master.memory", "8g")
    conf.set("spark.sql.shuffle.partitions",num_partitions)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    

    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    
    #Let the user choose the preferred pipeline
    print("There are three pipelines you can execute:")
    print("\t·Insert 1 to execute Data Management Pipeline")
    print("\t·Insert 2 to execute Data Analysis Pipeline")   
    print("\t·Insert 3 to execute Run-time Classifier Pipeline")  
    while True:
        pipeline = input("Enter pipeline code (enter 'q' to quit): ")
        if pipeline == "1":
            Data_Management_Pipeline.run(spark, URI_PATH_TO_DIRECTORY, num_partitions)
        elif pipeline == "2":
            if os.path.exists(URI_PATH_TO_DIRECTORY[11:]+"/data_training"):
                Data_Analysis_Pipeline.run(spark, URI_PATH_TO_DIRECTORY)
            else:
                print("There is no training data, you should execute first the Data Management Pipeline")
        elif pipeline == "3":
            if os.path.exists(URI_PATH_TO_DIRECTORY[11:]+"\dtc"):
                Run_time_Classifier_Pipeline.run(spark, URI_PATH_TO_DIRECTORY)
            else:
                print("There is no trained model, you should execute first the Data Analysis Pipeline")
        elif pipeline == "q":
            break
        else:
            print("Please insert a valid option")
        