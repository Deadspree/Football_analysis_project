from ultralytics import YOLO
model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save = True) #Save the output video of the result

print(results[0]) #First frame
print('=======================')
for box in results[0].boxes:
    print(box)