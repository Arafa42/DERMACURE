import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("./model/build")

BATCH_SIZE = 64
IMAGE_SIZE = 128

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset/skin_diseases/train",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size= BATCH_SIZE
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Dataset/skin_diseases/test",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

test_ds = test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[23].numpy().astype('uint8')
    first_label = labels_batch[25].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names[np.argmax(batch_prediction[23])])


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")
plt.show()