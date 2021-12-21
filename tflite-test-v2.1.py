print('TFLite LSTM Energy Classivation V2.1')

# imports
import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import cv2
import time
from datetime import datetime
import csv
import multiprocessing

# definitions

test_dataset_csv_path = 'gaf_data/gaf_dataset_test.csv'
test_dataset_path = 'gaf_data/test/'
files_list = []
labels = ['abnormal', 'normal']
predictions = []

# import data
edge_dataset = pd.read_csv(test_dataset_csv_path, header=0)
files_list = edge_dataset['image_name'].values
print(edge_dataset.head())

# prepare tflite model
tflite_model_eval_time_start = time.clock()
tflite_model_path = 'models/epochs-35-model.tflite'
interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Display inputs and outputs:')
print(input_details)
print(output_details)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('width:', width, 'height', height)

# metrics calculation function
def metrics(actual: np.array, predicted: np.array) -> np.array:
    # Confusion matrix elements
    TP = np.count_nonzero(predicted * actual)
    TN = np.count_nonzero((predicted - 1) * (actual - 1))
    FP = np.count_nonzero(predicted * (actual - 1))
    FN = np.count_nonzero((predicted - 1) * actual)

    # Classification evaluation metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return np.array([precision, recall, f1, accuracy])

# execute model (i.e. infer) on data files
for file in files_list:
  image_path = test_dataset_path + file 
  print('file:', image_path)
  image = cv2.imread(image_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  imH, imW, _ = image.shape 
  image_resized = cv2.resize(image_rgb, (width, height))
  input_data = np.expand_dims(image_resized, axis=0)

  # Perform the actual detection by running the model with the image as input
  interpreter.set_tensor(input_details[0]['index'],input_data)
  interpreter.invoke()

  prediction_tensor = interpreter.get_tensor(output_details[0]['index']) 
  prediction = not bool(np.argmax(prediction_tensor))
  prediction_label = labels[np.argmax(prediction_tensor)]
  predictions.append(prediction)
  print('TFLite tensor:', prediction_tensor, '- prediction label:', prediction_label, '- prediction:', prediction)
tflite_model_eval_time_elapsed = time.clock() - tflite_model_eval_time_start

# Calculate model evaluation metrics
_predictions = np.array(predictions)
_actual = edge_dataset['label_no'].values.astype('bool')
tflite_precision, tflite_recall, tflite_f1, tflite_accuracy = metrics(_actual, _predictions)
print('TFLite precision:', tflite_precision, 'recall:', tflite_recall, 'f1:', tflite_f1, 'accuracy:', tflite_accuracy)

# Print running stats
timestamp = str(datetime.now())
stats = f'''Run stats (in sec) - {timestamp}
---------------------------TFLite-----------------------
tflite_model_eval2_time_elapsed (using interpreter): {tflite_model_eval_time_elapsed} - accuracy: {tflite_accuracy}
-------------------------------------------------------
No. of CPU cores: {multiprocessing.cpu_count()}
'''
print(stats)

# tflite csv file header: ['timestamp', 'tflite_eval_time_elapsed', 'tflite_model_eval2_time_elapsed', 'tflite_precision', 'tflite_recall', 'tflite_f1', 'tflite_accuracy'])
stats_list = [timestamp, tflite_model_eval_time_elapsed, tflite_precision, tflite_recall, tflite_f1, tflite_accuracy]
with open('../stats_odroid.csv', "a") as csv_output_file:
    writer = csv.writer(csv_output_file)
    writer.writerow(stats_list)