# PyFaceDet
PyFaceDet is a Python wrapper of libfacedetection. This fork uses Python3 and matches the latest version of libfacedetection.

## Objective

PyFaceDet is a Python wrapper for quick face detection.

## Installation

- Clone
- Get the latest version of [libfacedetection](https://github.com/ShiqiYu/libfacedetection), compile
- Put the generated `libfacedetection.so` file in the directory of PyFaceDet.
- `pip3 install ./PyFaceDet`

## Usage

```python
  from PyFaceDet import facedetectcnn
  image = Image.open(...) # `Path`, PIL `Image`, `bytes`, and NumPy `Array` in BGR format are all compatible
  for x, y, w, h, confidence in facedetectcnn.facedetect_cnn(image):
      print(x, y, w, h, confidence)
      if confidence < 75: continue
      face = image.crop((x, y, x + w, y + h))
      yield face
```

Note: Parameters for `width`, `height` and `step` are necessary when use `bytes` object.
