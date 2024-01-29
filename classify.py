import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model
import cv2
image_size = (128, 128)


def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


content_type = 'image/jpeg'
headers = {'content-type': content_type}

model = load_model("fake_image_model.h5")
class_names = ["FAKE","REAL"]


def ela_image(image):
    img = convert_to_ela_image(image,90)
    img = np.array(img)
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded
    # response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

def predict_real_fake():
    img_path="images/fake1.png"
    image = prepare_image(img_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    
    print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

# im = cv2.imread(img_path)
# image = cv2.resize(im, (620,620))
# cv2.imshow("image",im)
# cv2.waitKey(0) & 0xFF

# cv2.imshow("image-ela",convert_to_ela_image(img_path,90))
# img = convert_to_ela_image(img_path,90)
# img = np.array(img)
# ela_image = cv2.resize(img, (620,620))

# cv2.imshow("image",im)

# cv2.waitKey(0) & 0xFF
# plt.plot()

# ret, jpeg = cv2.imencode('.jpeg', frame)
# return jpeg.tobytes()
