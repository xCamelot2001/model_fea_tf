import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from collections import defaultdict

# Constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64
EPOCHS = 25
TRAIN_DATA_DIR = './FER2013/train'
VALIDATION_DATA_DIR = './FER2013/test'
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to plot the distribution of data
def plot_data_distribution(train_counts, test_counts, labels):
    import matplotlib.pyplot as plt
    x = range(len(labels))
    fig, ax = plt.subplots()
    bar_width = 0.35
    rects1 = ax.bar(x, train_counts, bar_width, label='Train')
    rects2 = ax.bar([p + bar_width for p in x], test_counts, bar_width, label='Test')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Number of images')
    ax.set_title('Number of images per emotion')
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

# Counting the number of images per class
def count_images(data_dir):
    counts = {}
    for category in os.listdir(data_dir):
        path = os.path.join(data_dir, category)
        if os.path.isdir(path):
            counts[category] = len(os.listdir(path))
    return counts

train_counts = count_images(TRAIN_DATA_DIR)
test_counts = count_images(VALIDATION_DATA_DIR)
plot_data_distribution(list(train_counts.values()), list(test_counts.values()), list(train_counts.keys()))

# Data Augmentation to balance the dataset
def augment_data(data_dir, class_labels):
    image_counts = {class_label: len(os.listdir(os.path.join(data_dir, class_label))) for class_label in class_labels}
    max_images = max(image_counts.values())
    datagen_dict = {}

    for class_label in class_labels:
        if image_counts[class_label] < max_images:
            datagen_dict[class_label] = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
        else:
            datagen_dict[class_label] = ImageDataGenerator(rescale=1./255)

        if image_counts[class_label] < max_images:
            generator = datagen_dict[class_label].flow_from_directory(
                data_dir,
                classes=[class_label],
                target_size=(IMG_HEIGHT, IMG_WIDTH),
                color_mode='grayscale',
                batch_size=1,
                save_to_dir=os.path.join(data_dir, class_label),
                save_prefix='aug',
                save_format='png',
                shuffle=False)
            
            for _ in range(max_images - image_counts[class_label]):
                generator.next()

# Augment data for training dataset
augment_data(TRAIN_DATA_DIR, CLASS_LABELS)

# Model Definition
def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(len(CLASS_LABELS), activation='softmax'))

    return model

# Compiling the model
def compile_model(model):
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

# Preparing the data generators
def prepare_data_generators(train_dir, validation_dir, batch_size):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')
    
    return train_generator, validation_generator

# Training the model
def train_model(model, train_generator, validation_generator, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[early_stopping]
    )
    return history

# Save the trained model
def save_model(model, file_name):
    model.save(file_name)

# Putting it all together
def main():
    model = build_model((IMG_HEIGHT, IMG_WIDTH, 1))
    model = compile_model(model)
    train_generator, validation_generator = prepare_data_generators(TRAIN_DATA_DIR, VALIDATION_DATA_DIR, BATCH_SIZE)
    history = train_model(model, train_generator, validation_generator, EPOCHS, BATCH_SIZE)
    save_model(model, 'emotion_detection_model_v1.keras')

# Run the main function
if __name__ == '__main__':
    main()
