import argparse
import face_alignment

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import os, json, uuid
from shutil import copyfile

import hopenet
import utils
import photosources

argparser = argparse.ArgumentParser(description='Utility for automatic face annotations: 68 2D-points + head angles')
argparser.add_argument('--photos_dir', default='', help='Path to photos to process')
argparser.add_argument('--video_file', default='', help='Path to video to process')
argparser.add_argument('--angles_model', default='./models/hopenet_robust_alpha1.pkl',
                       help='Path to angles prediction model, \
                       download link can be found here: https://github.com/natanielruiz/deep-head-pose')
argparser.add_argument('--visualize', action='store_true', help='Show detection overlay')
argparser.add_argument('--output', default='./annoface', help='Where results should be saved')
argparser.add_argument('--maxdim', type=int, default=1000,
                       help='Max image dimension, images with greater values will be resized')
args = argparser.parse_args()

if not args.photos_dir and not args.video_file:
    print("Neither dir with photos or video file was passed as input! Abort...")
    exit(1)

if not os.path.exists(args.output):
    print(f"Output directory '{args.output}' will be created")
    os.makedirs(args.output)

print("Detectors initialization...")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')


print("Regressors initialization...")
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
cudnn.enabled = True
gpu = 0
model.load_state_dict(torch.load(args.angles_model))
transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model.cuda(gpu)
# Change model to 'eval' mode (BN uses moving mean/var).
model.eval()
idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

if args.photos_dir and args.video_file == '':
    photos_source = photosources.DirectorySource(args.photos_dir)
if args.video_file:
    photos_source = photosources.VideoSource(args.video_file)

while True:
    try:
        filename, image = photos_source.next()
        if image is not None:
            if filename is None:
                original_image = image.copy()
            print(f" - shape: {image.shape}")
            # as for the big pictures CUDA generates out of memory error
            scale = 1

            if image.shape[0] > args.maxdim or image.shape[1] > args.maxdim:
                scale = min(args.maxdim / image.shape[0], args.maxdim / image.shape[1])
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                print(f" - transformed to: {image.shape}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            results = fa.get_landmarks_from_image(rgb_image, return_bboxes=True)
            if results is None:
                print(f" - no faces found")
            else:
                landmarks = results[0]
                bboxes = results[1]
                print(f" - faces detected: {len(bboxes)}")
                annotations = []

                for i in range(len(bboxes)):
                    box = [int(item) for item in bboxes[i]]
                    # Angles prediction
                    x_min = box[0]
                    y_min = box[1]
                    x_max = box[2]
                    y_max = box[3]

                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width // 4
                    x_max += 2 * bbox_width // 4
                    y_min -= bbox_height // 4
                    y_max += bbox_height // 4
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(rgb_image.shape[1], x_max)
                    y_max = min(rgb_image.shape[0], y_max)
                    # Crop
                    pilimg = Image.fromarray(rgb_image[y_min:y_max, x_min:x_max])
                    # Transform
                    pilimg = transformations(pilimg)
                    pilimg_shape = pilimg.size()
                    pilimg = pilimg.view(1, pilimg_shape[0], pilimg_shape[1], pilimg_shape[2])
                    pilimg = Variable(pilimg).cuda(gpu)
                    # Inference
                    yaw, pitch, roll = model(pilimg)

                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)
                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                    print(f" - pitch: {pitch_predicted:.0f}, yaw: {yaw_predicted:.0f}, roll: {roll_predicted:.0f}")

                    # Visualization and serialization
                    if args.visualize:
                        utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted,
                                        tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=bbox_height / 2)

                    if args.visualize:
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(image, box[0:2], box[2:4], (0, 255, 0), 1, cv2.LINE_AA)
                    list_of_points = []
                    for pt in landmarks[i]:
                        list_of_points.append({"x": float(pt[0] / scale), "y": float(pt[1] / scale)})
                        if args.visualize:
                            cv2.circle(image, [int(pt[0]), int(pt[1])], 2, (255, 0, 127), 1, cv2.LINE_AA)
                    annotation = {
                        "rectangle": [float(bboxes[i][0] / scale), float(bboxes[i][1] / scale),
                                      float(bboxes[i][2] / scale), float(bboxes[i][3] / scale)],
                        "landmarks": list_of_points,
                        "pitch": float(pitch_predicted),
                        "yaw": float(yaw_predicted),
                        "roll": float(roll_predicted)
                    }
                    annotations.append(annotation)
                if args.visualize:
                    cv2.imshow('probe', image)
                    cv2.waitKey(0)

                unique_string = str(uuid.uuid4())
                if filename is not None:
                    image_filename = unique_string + filename.rsplit('.', 1)[1]
                    copyfile(os.path.join(args.photos_dir, filename),
                             os.path.join(args.output, image_filename))
                else:
                    image_filename = unique_string + ".jpg"
                    cv2.imwrite(os.path.join(args.output, image_filename), original_image)
                annotation_filename = unique_string + '.json'
                with open(os.path.join(args.output, annotation_filename), 'w') as of:
                    json.dump(annotations, of, indent=4)
        else:
            print(" - can not decode image")
    except StopIteration:
        break


'''if args.photos_dir:
    print(f"\nDirectory '{args.photos_dir}' will be processed")
    for filename in [f.name for f in os.scandir(args.photos_dir) if (f.is_file() and validextension(f.name))]:
        print(f"{filename}")
        image = cv2.imread(os.path.join(args.photos_dir, filename), cv2.IMREAD_COLOR)
        if image is not None:
            print(f" - shape: {image.shape}")
            # as for the big pictures CUDA generates out of memory error
            scale = 1

            if image.shape[0] > args.maxdim or image.shape[1] > args.maxdim:
                scale = min(args.maxdim / image.shape[0], args.maxdim / image.shape[1])
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                print(f" - transformed to: {image.shape}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            results = fa.get_landmarks_from_image(rgb_image, return_bboxes=True)
            if results is None:
                print(f" - no faces found")
            else:
                landmarks = results[0]
                bboxes = results[1]
                print(f" - faces detected: {len(bboxes)}")
                annotations = []

                for i in range(len(bboxes)):
                    box = [int(item) for item in bboxes[i]]
                    # Angles prediction
                    x_min = box[0]
                    y_min = box[1]
                    x_max = box[2]
                    y_max = box[3]

                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width // 4
                    x_max += 2 * bbox_width // 4
                    y_min -= bbox_height // 4
                    y_max += bbox_height // 4
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(rgb_image.shape[1], x_max)
                    y_max = min(rgb_image.shape[0], y_max)
                    # Crop
                    pilimg = Image.fromarray(rgb_image[y_min:y_max, x_min:x_max])
                    # Transform
                    pilimg = transformations(pilimg)
                    pilimg_shape = pilimg.size()
                    pilimg = pilimg.view(1, pilimg_shape[0], pilimg_shape[1], pilimg_shape[2])
                    pilimg = Variable(pilimg).cuda(gpu)
                    # Inference
                    yaw, pitch, roll = model(pilimg)

                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)
                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                    print(f"pitch: {pitch_predicted}, yaw: {yaw_predicted}, roll: {roll_predicted}")

                    # Visualization and serialization
                    if args.visualize:
                        utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted,
                                        tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size=bbox_height / 2)

                    if args.visualize:
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(image, box[0:2], box[2:4], (0, 255, 0), 1, cv2.LINE_AA)
                    list_of_points = []
                    for pt in landmarks[i]:
                        list_of_points.append({"x": float(pt[0] / scale), "y": float(pt[1] / scale)})
                        if args.visualize:
                            cv2.circle(image, [int(pt[0]), int(pt[1])], 2, (255, 0, 127), 1, cv2.LINE_AA)
                    annotation = {
                        "rectangle": [float(bboxes[i][0] / scale), float(bboxes[i][1] / scale),
                                      float(bboxes[i][2] / scale), float(bboxes[i][3] / scale)],
                        "landmarks": list_of_points,
                        "pitch": float(pitch_predicted),
                        "yaw": float(yaw_predicted),
                        "roll": float(roll_predicted)
                    }
                    annotations.append(annotation)
                if args.visualize:
                    cv2.imshow('probe', image)
                    cv2.waitKey(0)

                unique_string = str(uuid.uuid4())
                image_filename = unique_string + filename.rsplit('.', 1)[1]
                copyfile(os.path.join(args.photos_dir, filename),
                         os.path.join(args.output, image_filename))
                annotation_filename = unique_string + '.json'
                with open(os.path.join(args.output, annotation_filename), 'w') as of:
                    json.dump(annotations, of, indent=4)
        else:
            print(" - can not decode image")'''
