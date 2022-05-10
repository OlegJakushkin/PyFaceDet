# PyFaceDet
PyFaceDet is a Python wrapper of libfacedetection. This fork uses Python3 and matches the latest version of libfacedetection.

## Objective

PyFaceDet is a Python wrapper for quick face detection.

## Installation

Clone, dafault prebuilded libfacedetection is in PyFaceDet dir, install
```
!rm -rf ./libfacedetection
!rm -rf ./PyFaceDet

!git clone --recursive https://github.com/OlegJakushkin/libfacedetection
!cd libfacedetection && cmake -DCMAKE_INSTALL_PREFIX=/usr/ -DBUILD_SHARED_LIBS=ON  -DCMAKE_BUILD_TYPE=Release -DDEMO=OFF . &&    cmake --build . --config Release  && cmake --build . --config Release --target install

!git clone --recursive https://github.com/zhuth/PyFaceDet
!cp -fr /usr/lib/x86_64-linux-gnu/libfacedetection.* ./PyFaceDet/PyFaceDet/

!ls ./PyFaceDet/PyFaceDet/

!pip3 install ./PyFaceDet
!cp -fr /usr/lib/x86_64-linux-gnu/libfacedetection.* /usr/local/lib/python3.7/dist-packages/PyFaceDet/
```

## Usage

```python
from PyFaceDet import facedetectcnn
from PIL import Image, ImageDraw, ImageOps , ImageFilter

image = Image.open("img.jpg")
draw  = ImageDraw.Draw(image, "RGBA") 
for x, y, w, h, confidence in facedetectcnn.facedetect_cnn(image):
    print(x, y, w, h, confidence)
    if confidence < 75: continue
    face = image.crop((x, y, x + w, y + h))
    display(face)
    draw.rectangle([(x,y), (x +w,y + h)], fill=(200, 100, 0, 127), outline ="red")

display(image)
```

![image](https://user-images.githubusercontent.com/2915361/167689798-b282fc10-5cb3-4398-ae21-a7e997966ed4.png)

Note: Parameters for `width`, `height` and `step` are necessary when use `bytes` object.
