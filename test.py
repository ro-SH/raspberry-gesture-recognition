import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam

MODEL_NAME = "my_model.h5"
IMAGES_PATH = "./images"

gesture_names = {
    0: "okay",
    1: "thumbs up",
    2: "v sign"
}

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Hold the background frame for background subtraction.
background = None
BG_WEIGHT = 0.5

region_top = 0
region_bottom = FRAME_HEIGHT * 2 // 3
region_left = FRAME_WIDTH // 2
region_right = FRAME_WIDTH

def prepare_image(img):
    img = np.stack((img,)*3, axis=-1)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    return img

def predict_image(model, img):
    img = np.array(img, dtype='float32')
    img /= 255
    pred_array = model.predict(img)

    # model.predict() returns an array of probabilities - 
    # np.argmax grabs the index of the highest probability.
    result = gesture_names[np.argmax(pred_array)]
    
    # A bit of magic here - the score is a float, but I wanted to
    # display just 2 digits beyond the decimal point.
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def get_model():

    try:
        return load_model(MODEL_NAME)
    except OSError:
        pass

    base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(3,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    train_generator=train_datagen.flow_from_directory(IMAGES_PATH,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10)
    model.save(MODEL_NAME)

    return model

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def get_region(img):
    # Separate the region of interest from the rest of the frame.
    region = img[region_top:region_bottom, region_left:region_right]
    # Make it grayscale so we can detect the edges more easily.
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Use a Gaussian blur to prevent frame noise from being labeled as an edge.
    region = cv2.GaussianBlur(region, (5,5), 0)

    return region

def get_average(region):
    # We have to use the global keyword because we want to edit the global variable.
    global background
    # If we haven't captured the background yet, make the current region the background.
    if background is None:
        background = region.copy().astype("float")
        return
    # Otherwise, add this captured frame to the average of the backgrounds.
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(image, threshold=25):
    global background
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

# Main function
def main():

    is_on = False
    num_frames = 0
    text = ''

    model = get_model()

    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        img = cv2.flip(img, 1)
        key = cv2.waitKey(10)
        if key == 27:
            break
        if key == 32:
            is_on = not is_on

        region = get_region(img)
        if num_frames < 30:
            get_average(region)
        else:
            hand = segment(region)
            if is_on and num_frames % 30 == 0:
                if hand is not None:
                    (thresholded, segmented) = hand

                    preprocessed_image = prepare_image(thresholded)
                    result, score = predict_image(model, preprocessed_image)
                    text = f'{result}, {score}%'
                else:
                    text = ''
        
        cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
        cv2.putText(img, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(img, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)

        num_frames += 1

        cv2.imshow('mycam', img)
    
    cam.release()
    cv2.destroyAllWindows()

main()