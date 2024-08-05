import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization

#generators
train_ds=keras.utils.image_dataset_from_directory(
directory='PetImages\Train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)

test_ds=keras.utils.image_dataset_from_directory(
directory='PetImages\Test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
)


def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = test_ds.map(process)

#DATA_SET HANDLING(REMOVAL OF UNUSUAL FILES)
#removal of 2 channel files
from PIL import Image
import os

# Define the path to the parent directory
parent_dir = r'C:\myamu.ac.in\Desktop\DEEP LEARNING\CNN\PetImages'  # Use raw string to avoid issues with backslashes
sub_dirs = ['Train', 'Test']
classes = ['Cat', 'Dog']

# Function to check if an image has an unexpected number of channels
def has_unexpected_channels(image_path):
    try:
        img = Image.open(image_path)
        mode = img.mode
        if mode not in ['L', 'RGB', 'RGBA']:  # 'L' for grayscale, 'RGB' for 3 channels, 'RGBA' for 4 channels
            print(f"Image {image_path} has unexpected mode: {mode}")
            return True
        return False
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return True  # Treat as corrupted if an error occurs

# Iterate through all images, check for unexpected channels, and remove those images
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(parent_dir, sub_dir)
    if not os.path.exists(sub_dir_path):
        print(f"Directory does not exist: {sub_dir_path}")
        continue
    for cls in classes:
        class_folder = os.path.join(sub_dir_path, cls)
        if not os.path.exists(class_folder):
            print(f"Directory does not exist: {class_folder}")
            continue
        print(f"Checking directory: {class_folder}")  # Debug print statement
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if not os.path.exists(file_path):
                print(f"File does not exist, removing reference: {file_path}")
                os.remove(file_path)
                continue
            if has_unexpected_channels(file_path):
                print(f"Removing image with unexpected channels: {file_path}")
                os.remove(file_path)

print("Check and removal of images with unexpected channels completed.")


 #REmoval of corrupted files
from PIL import Image
import os

# Define the path to the parent directory
parent_dir = r'C:\myamu.ac.in\Desktop\DEEP LEARNING\CNN\PetImages'  # Use raw string to avoid issues with backslashes
sub_dirs = ['Train', 'Test']
classes = ['Cat', 'Dog']

# Function to check if an image is corrupted
def is_image_corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return False
    except Exception as e:
        print(f"Image {image_path} is corrupted: {e}")
        return True

# Iterate through all images, check for corruption, and remove corrupted ones
for sub_dir in sub_dirs:
    for cls in classes:
        class_folder = os.path.join(parent_dir, sub_dir, cls)
        print(f"Checking directory: {class_folder}")  # Debug print statement
        if not os.path.exists(class_folder):
            print(f"Directory does not exist: {class_folder}")
            continue
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if not os.path.exists(file_path):
                print(f"File does not exist, removing reference: {file_path}")
                os.remove(file_path)
                continue
            if is_image_corrupted(file_path):
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

print("Check and removal of corrupted images completed.")

 #removal of Unsupported format image
from PIL import Image
import os

# Define the path to the parent directory
parent_dir = r'C:\myamu.ac.in\Desktop\DEEP LEARNING\CNN\PetImages'  # Use raw string to avoid issues with backslashes
sub_dirs = ['Train', 'Test']
classes = ['Cat', 'Dog']

# Function to check if an image has an acceptable format
def is_supported_format(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, in fact, an image
        return img.format in ['JPEG', 'PNG', 'GIF', 'BMP']
    except Exception as e:
        print(f"Error verifying image {image_path}: {e}")
        return False  # Treat as unsupported if an error occurs

# Iterate through all images, check for unsupported formats, and remove those images
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(parent_dir, sub_dir)
    if not os.path.exists(sub_dir_path):
        print(f"Directory does not exist: {sub_dir_path}")
        continue
    for cls in classes:
        class_folder = os.path.join(sub_dir_path, cls)
        if not os.path.exists(class_folder):
            print(f"Directory does not exist: {class_folder}")
            continue
        print(f"Checking directory: {class_folder}")  # Debug print statement
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            if not os.path.exists(file_path):
                print(f"File does not exist, removing reference: {file_path}")
                os.remove(file_path)
                continue
            if not is_supported_format(file_path):
                print(f"Removing unsupported image: {file_path}")
                os.remove(file_path)

print("Check and removal of unsupported images completed.")
 
#OPTIMIZATION
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(train_ds,validation_data=test_ds,epochs=6)