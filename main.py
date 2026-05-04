from ultralytics import YOLO
import cv2

# 1. Load the YOLO model
yolo = YOLO("yolov8n.pt")

# 2. Load the image
img = cv2.imread("object.png")
if img is None:
    raise ValueError("Image 'object.png' not found.")

# 3. Run YOLO inference directly on the image
results = yolo(img)

# to get resolution
for r in results:
    image_height, image_width = r.orig_shape

# print(image_height, image_width)

with open("data", "r") as f:
    contents = f.read().split()
del contents[0]
# print(contents)

x_center = float(contents[0])
y_center = float(contents[1])
box_w = float(contents[2])
box_h = float(contents[3])

px_x = x_center * image_width
px_y = y_center * image_height
px_w = box_w * image_width
px_h = box_h * image_height

px_x = max(0, image_width)
px_y = max(0, image_height)
px_w = min(0, image_width)
px_h = min(0, image_height)

x1 = int(px_x - px_w / 2) #<-left edge
y1 = int(px_y - px_h / 2) #<-top edge
x2 = int(px_x + px_w / 2) #<-right edge
y2 = int(px_y + px_h / 2) #<-bottom edge

x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(image_width, x2)
y2 = min(image_height, y2)

crop = img[y1:y2, x1:x2]

if crop.size > 0:
    # save and display
    cv2.imshow("annoated-image", crop)
    cv2.waitKey(0)
    cv2.imwrite("annoated-image", crop)
else:
    print("Crop failed: result was 0 pixels.")