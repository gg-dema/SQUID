import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_contour(image_path, num_bundle=3, kernel_dimension=13):
  image = cv2.imread(image_path)

  # Rotate the image by 90 degrees (or any angle you prefer)
  angle = 180
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
  rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
  image = rotated_image

  # convert to gray scale and build a mask
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

  # Structuring element
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dimension, kernel_dimension))

  # Original contour (binary image gives filled region)
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  main_contour = max(contours, key=cv2.contourArea)

  # Generate external (dilated) contours
  dilated = binary.copy()
  dilated_contours = []
  for i in range(1, num_bundle):  # Generate 3 exterior layers
      dilated = cv2.dilate(dilated, kernel, iterations=1)
      contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      dilated_contours.append(max(contours, key=cv2.contourArea))


  # Generate internal (eroded) contours
  eroded = binary.copy()
  eroded_contours = []
  for i in range(1, num_bundle):  # Generate 1 interior layer
      eroded = cv2.erode(eroded, kernel, iterations=1)
      contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      if contours:
          eroded_contours.append(max(contours, key=cv2.contourArea))

  # convert the eroded and dilated contour to np.array, then stack it
  dilated_contours, eroded_contours = np.concatenate(dilated_contours), np.concatenate(eroded_contours)

  return main_contour, np.concatenate([dilated_contours, eroded_contours])



image = "batman.jpg"
main, hard_neg = extract_contour(image,
                                 kernel_dimension=20, # ps: larger kernel, larger distance between bundle
                                 num_bundle=5)        # bundle inside and outside --> num_bundle * 2
np.save(f"{image}_main.npy", main)
np.save(f"{image}_hard_neg.npy", hard_neg)

plt.plot(main[:, 0, 0], main[:, 0, 1])
plt.plot(hard_neg[:, 0, 0], hard_neg[:, 0, 1])
plt.show()
