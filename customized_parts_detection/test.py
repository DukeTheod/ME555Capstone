import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/theod/Documents/ME555Capstone/object_detection/yolov5/runs/train/exp5/weights/last.pt')
im = '/home/theod/Documents/ME555Capstone/object_detection/dataset/train/images/image5.jpg'
results = model(im)
results.print()
results.show()
print(f'result is here: {results.pandas().xyxy}')