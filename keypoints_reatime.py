import matplotlib
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

matplotlib.use('Qt5Agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image = letterbox(cap.read()[1], (int(cap.get(3))), stride=64, auto=True)[0]

    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)  
    
    with torch.no_grad():
        output, _ = model(image)

    



    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)
    

    


    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)

        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)

        cv2.rectangle(

            nimg,

            (int(xmin), int(ymin)),

            (int(xmax), int(ymax)),

            # color=(255, 0, 0),
            color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),

            thickness=1,

            lineType=cv2.LINE_AA

        )


    cv2.imshow('live', nimg)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()