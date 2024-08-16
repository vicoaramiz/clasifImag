
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Carga y preprocesa CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Muestra 16  entrenamientos a 4x4 grid
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# Liimita los datos para un entrenamiento rapido (optional)
training_images, training_labels = training_images[:20000], training_labels[:20000]
testing_images, testing_labels = testing_images[:4000], testing_labels[:4000]

# Construye el  CNN modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compila el modelo
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Entrena el modelo
model.fit(training_images, training_labels, epochs=10, 
          validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')

# Carga y preprocesa la imagen por prediccion
def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert to RGB
    img = cv.resize(img, (32, 32))  # Resize to match model input
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and add batch dimension
    return img

# Predice y muestra los resultados
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    plt.imshow(img[0])
    plt.title(f'Prediction: {class_names[index]}')
    plt.show()
    print(f'Prediction is {class_names[index]}')

# Test con imagenes
predict_image('caballo.jpg')
predict_image('deer.jpg')
predict_image('plane.jpg')
