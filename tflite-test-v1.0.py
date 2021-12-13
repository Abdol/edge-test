print('Hi')

import tflite_runtime.interpreter as tflite

tflite_model_path = 'model.tflite'
interpreter = tflite.Interpreter(model_path=tflite_model_path)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Display inputs and outputs:')
print(input_details)
print(output_details)

import cv2
image_path = 'daisy.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape 
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Perform the actual detection by running the model with the image as input
interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index']) # Bounding box coordinates of detected obj$
print(boxes)
