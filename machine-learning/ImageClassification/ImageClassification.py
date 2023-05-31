import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data_dir_train = './assets/train'
data_dir_val = './assets/validation'
data_dir_test = './assets/test'

batch_size = 300
img_height = 180
img_width = 180

def ModelTrainAndSave():
    # Load training dataset
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir_train,
      validation_split=0.8,
      subset="both",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    # Load validation dataset
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #   data_dir_val,
    #   validation_split=0.2,
    #   subset="validation",
    #   seed=123,
    #   image_size=(img_height, img_width),
    #   batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    # show datasets 
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    # show
    # plt.show()

    for image_batch, labels_batch in train_ds:
      print(image_batch.shape)
      print(labels_batch.shape)
      break

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Stardartize the data

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # Create a model
    num_classes = len(class_names)

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.summary()

    # Train the model
    epochs=10
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    # Visualize training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.show()

    # ----------------------------------Overfitting------------------------------------------

    # Data augmentation
    # Overfitting generally occurs when there are a small number of training examples. 
    # Data augmentation takes the approach of generating additional training data from your 
    # existing examples by augmenting them using random transformations that yield believable-looking images.
    # This helps expose the model to more aspects of the data and generalize better.
    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                      img_width,
                                      3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
      ]
    )

    # Visualize a few augmented examples by applying data augmentation to the same image several times:
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
      for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
    # plt.show()


    # -----------------------------------Dropout ----------------------------------------------------

    # Another technique to reduce overfitting is to introduce dropout regularization to the network.
    # When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) a number of
    #  output units from the layer during the training process. Dropout takes a fractional number as its input 
    # value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units 
    # randomly from the applied layer.

    # Create a model
    model = tf.keras.Sequential([
      data_augmentation,
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes, name="outputs")
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # model.summary()

    # Train the model
    epochs = 15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    # Visualize training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Save the model
    model.save('./workspace/sift2_classification_model.h5')

# -------------------------Predict on new data img --------------------------------------
def LoadAndPredict():
    # Load the saved model
    loaded_model = tf.keras.models.load_model("./workspace/sift2_classification_model.h5")

    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_val,
    validation_split=0.8,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = test_ds.class_names

    for img, lbl in test_ds.take(1):
        for i in range(10):
            img_array = tf.keras.utils.img_to_array(img[i].numpy().astype("uint8"))
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = loaded_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence"
                .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

    # for path in test_ds.file_paths:
    #     img = tf.keras.utils.load_img(
    #         path, target_size=(img_height, img_width)
    #     )
    #     img_array = tf.keras.utils.img_to_array(img)
    #     img_array = tf.expand_dims(img_array, 0) # Create a batch

    #     predictions = loaded_model.predict(img_array)
    #     score = tf.nn.softmax(predictions[0])

    #     print(
    #         "This image most likely belongs to {} with a {:.2f} percent confidence."
    #         .format(class_names[np.argmax(score)], 100 * np.max(score))
    #     )

if __name__ == '__main__':
    print("Type your choice: [1: Model Train and Save, 2: Model Load and Predict]")
    choice = input()
    if int(choice) == 1:
        ModelTrainAndSave()
    else:
        LoadAndPredict()