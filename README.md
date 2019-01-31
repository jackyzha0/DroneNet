# DroneNet
### Jacky Zhao - Science Fair Project 2018-19

## Summary
Decentralized drone swarm communication for search and rescue missions

Problems with current methods:
Heavily reliant on global communication methods such as GPS and central communication unit

## Implementation
1. Hardware
  - Positioning
    - PXFmini
      - 3D Gyroscope
      - Barometer
      - 3D Compass
  - Flying
    - Power
      - LiPoly 1300mAh 4C
    - Thrust
      - 5045GF with RS2205 Motor
    - Drone Frame
      - 3D Printed
  - Communications
    - Low Power Bluetooth
      - nRF52832
2. Software
  - Decentralized Autopilot with reinforcement learning
    - input array: [n, r, phi, theta] where n is number of total drones, r is distance away in pixels, phi is azimuth angle, and theta is elevation angle.
    - r is assumed to be infinity when out of range
    - Cover area in smallest time possible while maintaining communication distance
    - Cost function should be affected by total flight time and integral of the distance between drones
    - Long-Short-Term-Memory Cell Network with outputs as velocity deltas (shape [dx, dy, dz])
  - Visualization
    - Python script to visualize position and direction of drones in drone swarm
    - Training environment for autopilot
    - Drone physics
      - Acceleration
      - Gravity
      - Bounding boxes and collisions
      - Air Resistance
  - Object Detection
    - YOLOv1 with modified class output, input size [448, 448]
    - Dataset
      - KITTI Vision Benchmark
      - [WIP] COCO Dataset
  - Hardware interfacing

## Temporary Notes Section
Free GPIO pins:
RPI_GPIO5
RPI_GPIO6
RPI_GPIO12
RPI_GPIO13
RPI_GPIO16
RPI_GPIO20
RPI_GPIO21
RPI_GPIO22
RPI_GPIO26

## TODO
- [ ] Check for remote sensing datasets
- [ ] Look at IR Cameras
- [ ] Get location from PXFMini
- [ ] Evaluate types of machine learning (look into DQNs)
- [ ] Create training scenario
- [x] Fix documentation for project
- [x] Setup hotspot on laptop
- [x] Configure Erle PXFMini
- [x] Build and compile APM for PXFMini and Pi0
- [x] Buy USB Wifi Dongle
- [x] Buy micro-HDMI to HDMI Adapter
- [x] Visualize convolutional kernels
- [x] Reimplement batch norm
- [x] Add YOLOv1 Full Architecture
- [x] Attempt restore weights
- [x] Switch to dropout
- [x] Switch to tiny-YOLO architecture
- [x] Make validation testing more frequent
- [x] Validation time image saving
- [x] Refresh dataset
- [x] Add documentation for detectionNet
- [x] Reshape func
- [x] Define TP / TN / FP / FN
- [x] Fix non-max suppression
- [x] Calculate IOU
- [x] Ensure weight saving saves current epoch
- [x] Add test evaluation
- [x] Add checkpoint saving
- [x] Add ability to restore weights
- [x] Add augmentation after read
- [x] np Pickle Writer
- [x] Implement data augmentation
- [x] Add batch norm
- [x] Interpret output with confidence filtering
- [x] Linearize last layer
- [x] Normalize xy
- [x] Normalize wh
- [x] Add label output in Tensorboard
- [x] Add checkpoint saving
- [x] Add FC Layers
- [x] Make sure capable of overfitting
- [x] imshow Output
- [x] Interpret Output
- [x] Check labels for accuracy
- [x] Check values of init labels
- [x] Update descriptions about dataset
- [x] Update software description in README
- [x] Calibrate PiCamera (Completed)
- [x] Solder motors to ESCs (Completed Nov. 1st)
- [x] Screw in new camera and PDB mount  (Completed Nov. 1st)
- [x] Install new propellers (Completed Oct. 31st)
- [x] Write function to calibrate camera intrinsic matrix (Completed Oct. 31st)
- [x] Implement photo functionality into camera class (Completed Oct. 31th)
- [x] Design camera mount (Completed Oct. 29th)
- [x] Design power distributor board model (Completed Oct. 30th)
- [x] Recut and redrill carbon fiber tubes (Completed Oct. 26th)
- [x] Reorder 2 propellers and LiPo charger (Completed Oct. 27th)
- [x] Add Camera class and integrate into Drone class (Completed Oct. 23th)
- [x] Add directionality to Drone class (Completed Oct. 23th)
- [x] Evaluate plausibility of homogenous coordinate system (Completed Oct. 22th)
- [x] Cut and drill carbon fiber tubes (Completed Oct. 19th)
- [x] Finish Drone Summary (Completed Oct. 19th)
- [x] Draw Circuits (Completed Oct. 19th)
- [x] Explain details of motors, thrust, etc. (Completed Oct. 19th)
- [x] Update Printing Logs (Completed Oct. 18th)
- [x] Find Thrust and Voltages required given current motors (Completed Oct. 18th)
- [x] Decide on Power Supply (Completed Oct. 18th)
- [x] Determine right electronic speed controller (ESC) to buy (Completed Oct. 18th)
- [x] Print Arms (Completed Oct. 18th)
- [x] Calculate Approximate Weight of each Drone (Completed Oct. 17th)
- [x] Print Carbon Fiber Jigs (Completed Oct.17th)
- [x] Print Bumpers (Completed Oct. 16th)
- [x] Print Sides (Completed Oct. 16th)
- [x] Print Lower Plate (Completed Oct. 16th)
- [x] Print Top Plate (Completed Oct. 16th)
- [x] Order PXFMini parts (Completed Oct. 15th)
- [x] Order Firefly Parts (Completed Oct. 12th)
- [x] Realtime rendering of 3D Space (Completed Oct. 2)
- [x] Particle based display for drone (Completed Sept. 28)
- [x] Create initial variables for drones (Completed Sept. 26)
- [x] Create drone class (Completed Sept. 26)

## Network Debug Notes
##### Network interfaces
wlx4cedfbb833a6 -- Asus AC56R Wifi Adapter (rtl8812au driver) -- Used for WiFi Access

wlp2s0 -- Builtin Intel Wifi Adapter -- Used for creating wireless access point

##### Debug Commands

nmcli device status -- lists network interfaces and status

dmesg -- general debug

arp -a -- Lists all devices connected to hotspot

service network-manager restart -- restarts wifi service

ssh pi@10.42.0.74

## detectionNet Explained
A modified version of YOLOv1 is implemented

Dimensionality of the next layer can be computed as follows:
![fig. 1](http://mathurl.com/ybw6b7yt.png)

3 Different models have been implemented as follows:

1) SqueezeNet

| Layer | Output Dimensions | Filter Size | Stride | Depth |
|-------|------------|-------------|--------|-------|
| images | [448, 448, 3] | - | - | - |
| conv1 | [224, 224, 96] | [7, 7] | [2, 2] | 96 |
| maxpool1 | [112, 112, 96] | [3, 3] | [2, 2] | - |
| fire1 | [112, 112, 64] | - | - | 64 |
| fire2 | [112, 112, 64] | - | - | 64 |
| fire3 | [112, 112, 128] | - | - | 128 |
| maxpool2 | [56, 56, 128] | [3, 3] | [2, 2] | - |
| fire4 | [56, 56, 128] | - | - | 256 |
| fire5 | [56, 56, 192] | - | - | 384 |
| fire6 | [56, 56, 192] | - | - | 384 |
| fire7 | [56, 56, 256] | - | - | 512 |
| maxpool3 | [28, 28, 256] | [3, 3] | [2, 2] | - |
| fire8 | [28, 28, 256] | - | - | 512 |
| avgpool1 | [12, 12, 256] | [7, 7] | [2, 2] | - |
| fc1 | [1024] | - | - | -1 (flatten) |
| fc2 | [4096] | - | - | - |
| fc3 | [675] | - | - | - |

2) Tiny YOLOv1

| Layer | Output Dimensions | Filter Size | Stride | Depth |
|-------|------------|-------------|--------|-------|
| images | [448, 448, 3] | - | - | - |
| conv1 | [448, 448, 16] | [3, 3] | [1, 1] | 16 |
| maxpool1 | [224, 224, 16] | [2, 2] | [2, 2] | - |
| conv2 | [224, 224, 32] | [3, 3] | [1, 1] | 32 |
| maxpool2 | [112, 112, 32] | [2, 2] | [2, 2] | - |
| conv3 | [112, 112, 64] | [3, 3] | [1, 1] | 64 |
| maxpool3 | [56, 56, 64] | [2, 2] | [2, 2] | - |
| conv4 | [56, 56, 128] | [3, 3] | [1, 1] | 128 |
| maxpool4 | [28, 28, 128] | [2, 2] | [2, 2] | - |
| conv5 | [28, 28, 256] | [3, 3] | [1, 1] | 256 |
| maxpool5 | [14, 14, 256] | [2, 2] | [2, 2] | - |
| conv6 | [14, 14, 512] | [3, 3] | [1, 1] | 512 |
| maxpool6 | [7, 7, 512] | [2, 2] | [2, 2] | - |
| conv7 | [7, 7, 1024] | [3, 3] | [1, 1] | 1024 |
| conv8 | [7, 7, 256] | [3, 3] | [1, 1] | 256 |
| conv9 | [7, 7, 512] | [3, 3] | [1, 1] | 512 |
| fc1 | [1024] | - | - | -1 (flatten) |
| fc2 | [4096] | - | - | - |
| fc3 | [675] | - | - | - |

3) YOLOv1

| Layer | Output Dimensions | Filter Size | Stride | Depth |
|-------|------------|-------------|--------|-------|
| images | [448, 448, 3] | - | - | - |
| conv1 | [224, 224, 64] | [7, 7] | [2, 2] | 64 |
| maxpool1 | [112, 112, 64] | [2, 2] | [2, 2] | - |
| conv2 | [112, 112, 192] | [3, 3] | [1, 1] | 192 |
| maxpool2 | [56, 56, 192] | [2, 2] | [2, 2] | - |
| conv3 | [56, 56, 128] | [1, 1] | [1, 1] | 128 |
| conv4 | [56, 56, 256] | [3, 3] | [1, 1] | 256 |
| conv5 | [56, 56, 256] | [1, 1] | [1, 1] | 256 |
| conv6 | [56, 56, 512] | [3, 3] | [1, 1] | 512 |
| maxpool3 | [28, 28, 512] | [2, 2] | [2, 2] | - |
| conv7 | [28, 28, 256] | [1, 1] | [1, 1] | 256 |
| conv8 | [28, 28, 512] | [3, 3] | [1, 1] | 512 |
| conv9 | [28, 28, 256] | [1, 1] | [1, 1] | 256 |
| conv10 | [28, 28, 512] | [3, 3] | [1, 1] | 512 |
| conv11 | [28, 28, 256] | [1, 1] | [1, 1] | 256 |
| conv12 | [28, 28, 512] | [3, 3] | [1, 1] | 512 |
| conv13 | [28, 28, 256] | [1, 1] | [1, 1] | 256 |
| conv14 | [28, 28, 512] | [3, 3] | [1, 1] | 512 |
| conv15 | [28, 28, 512] | [1, 1] | [1, 1] | 512 |
| conv16 | [28, 28, 1024] | [3, 3] | [1, 1] | 1024 |
| maxpool4 | [14, 14, 1024] | [2, 2] | [2, 2] | - |
| conv17 | [14, 14, 512] | [1, 1] | [1, 1] | 512 |
| conv18 | [14, 14, 1024] | [3, 3] | [1, 1] | 1024 |
| conv19 | [14, 14, 512] | [1, 1] | [1, 1] | 512 |
| conv20 | [14, 14, 1024] | [3, 3] | [1, 1] | 1024 |
| conv21 | [14, 14, 1024] | [3, 3] | [1, 1] | 1024 |
| conv22 | [7, 7, 1024] | [3, 3] | [2, 2] | 1024 |
| conv23 | [7, 7, 1024] | [3, 3] | [1, 1] | 1024 |
| conv24 | [7, 7, 1024] | [3, 3] | [1, 1] | 1024 |
| fc2 | [4096] | - | - | -1 (flatten) |
| fc3 | [675] | - | - | - |

Batch normalization is applied after every conv and fc layer before leaky ReLu

Input Format:
img - [batchsize, 448, 448, 3]

labels - [batchsize, sx, sy, B * (C + 4)] where B is number of bounding boxes per grid cell (3) and C is number of classes (4)

Note: Form [p1x, p1y, p2x, p2y] have been converted to form [x,y,w,h]

##### Cost Function
Two constants are set to correct the unbalance between obj and no_obj boxes, ![constants](http://mathurl.com/yaafekee.png)


YOLO works by adding the following seperate losses, ![t_loss](http://mathurl.com/yc2sghmx.png)

Where,

lossXYWH is the sum of the squared errors of the x,y,w,h values of bounding boxes for all squares and bounding boxes responsible

![eqn1](http://mathurl.com/yb8r7ll2.png)

lossObjConf is the sum of the squared errors in predicted confidences of all bounding boxes with objects

![eqn2](http://mathurl.com/y6wja6g8.png)

lossNoobjConf is the sum of the squared errors in predicted confidences of all grid cells with no objects

![eqn3](http://mathurl.com/ybs83qun.png)

lossProb is the sum of the squared errors in predicted class probabilities across all grid cells

![eqn3](http://mathurl.com/ybaezuc2.png)

Variable Definitions:
phi is defined as ![phidef](http://mathurl.com/yc2gtzso.png)

sx, sy are defined as number of horizontal grid cells and vertical grid cells, respectively
classes is the list of total possible classes
B is the number of bounding boxes per grid cell

##### Fire Module
A fire module (described in SqueezeNet) is defined as 3 1x1 conv2d layers followed by 4 1x1 conv2d layers concatenated with 4 3x3 conv2d layers

## Train Details
Tensorflow version == 1.2.0

GPU == NVIDIA 970M

Batchsize == 4

Learning Rate == 1e-3

Optimizer == Adam Optimizer

Adam Epsilon == 1e-0

Batch Normalization Momentum == 0.9

Batch Normalization Epsilon == 1e-5

sx == 7

sy == 7

B == 3

C == 4

Alpha (Leaky ReLu coefficient) = 0.1

## Data formatting of the KITTI Vision Dataset
### Images
Of format 000000.png <br/>
Training set size: ~7500 images
Testing set size: ~7500 images
Various dimensions from [1224, 370] to [1242, 375]
##### Processing
Images
1. Resizing all uniform [1242, 375] with cubic interpolation
2. Crop random to [375, 375]
3. Normalize to range [0., 1.]
4. Pad size to [448, 448]

Labels
1. Discard all labels that are not Car / Pedestrian / Cyclist / Misc. Vehicle (Truck or Van)
2. Discard all labels outside crop range
3. One-hot encode labels
4. Convert p1p2 to xywh
5. Assign boxes to cells
5. Normalize w,h to cell dimensions
6. Normalize x,y to image dimensions
7. Append obj, no_obj, objI boolean masks

### Annotations
All annotations are white-space delimited <br/>
__Object Annotations__ <br/>
Of format 000000.txt <br/>
15 Columns of separated data
-1 is default value if field does not apply
1. Class - Car, Van, Truck, Pedestrian, Person Sitting, Cyclist, Tram, Misc., Don't Care
2. Truncated - If the bounding box is truncated (leaves the screen)
3. Occluded - If the bounding box/class is partially obscured
4. Observation angle - Range from -pi to pi
5. Bounding box x_min
6. Bounding box y_min
7. Bounding box x_max
8. Bounding box y_max
9. 3D - x dimension of object
10. 3D - y dimension of object
11. 3D - z dimension of object
12. x location
13. y location
14. z location
15. ry rotation around y-axis

For simplicity sake, only columns 1,5,6,7,8 will be used.

## Price and Weight Data
| Qty | Item | Total Weight (g) | Price |
|-----|------|------------------|-------|
| 1x | 3D Printed Frame | 50.00g | N/A |
| 1x | LiPo Battery | 165.00g | $19.10 |
| 4x | Electronic Speed Controller | 28.00g | $50.68 |
| 1x | Power Distribution Board | 19.30g | $4.13 |
| 4x | 100mm Carbon Fiber Tubes | 45.20g | $7.99 |
| 1x | PXFmini Power Module | 50.00g | $44.94 |
| 1x | PXFmini | 15.00g | $103.36 |
| 4x | Motors | 120.00g | $39.96 |
| 4x | Propellers | 21.2g | $8.76 |
| 1x | Pi Zero | 9.00g | $5.00 |
| 1x | Pi Camera | 18.14g | $14.99 |
| N/A | Misc. Wires | 20.0g | N/A |
| Totals | N/A | 560.84g | $298.91 |

## Drone Stats
__Battery:__ 1300mAh at 45C. Max recommended current draw = Capacity (Ah) * C Rating = 1.3*45 = ___58.5A___<br/>
__Thrust Required:__ Because the drone is not intended for racing, a thrust to weight ratio of ___4:1___ works well. A total thrust of ___2.2433kg___ is required, meaning ___560g___ of thrust per motor.<br/>
__Current:__ Looking at the thrust table below, we can see that 13A @ 16.8V on a RS2205-2300KV with GF5045BN Propellers nets us almost exactly 560g of thrust. Multiplying the current for each motor, we end up with a maximum total current of __52A__, well below the 58.5 theoretical maximum of the LiPo battery used.<br/>
__Flight Time:__ We can find the current for which all motors provide enough thrust to keep the drone in the air (560g total or 140g each). Looking at the thrust table, we can see that a little less than 3A nets us around 140g of thrust. Assuming only 80% of capacity is effective, we 1.04mAh available. Multiplying by h/60min and dividing by the current draw 3A, we get a theoretical flight time of 20.8 minutes.

## Thrust Table for RS2205-2300KV @ 16.8V with GF5045BN
| Current (A) | Thrust (g) | Efficiency (g/W) | Speed (RPM) |
|-------------|------------|------------------|-------------|
| 1 | 76 | 4.75 | 7220 |
| 3 | 183 | 3.81 | 10790 |
| 5 | 282 | 3.54 | 13030 |
| 7 | 352 | 3.10 | 14720 |
| 9 | 426 | 2.93 | 16180 |
| 11 | 497 | 2.82 | 17150 |
| 13 | 560 | 2.69 | 18460 |
| 15 | 628 | 2.62 | 19270 |
| ... | ... | ... | ... |
| 27 | 997 | 2.28 | 23920 |
| 30 | 1024 | 2.14 | 24560 |

![Rough Diagram of circuit](Logs/fig1.jpg?raw=true "Circuit Diagram ")
Rough Diagram of circuit.

## Materials
- [x] 4x 3pin motor to ESC female to female connector
- [x] 4x M3 Locknuts
- [x] 2x M3x22mm spacers
- [x] 4x M3x32mm
- [x] 4x M3 Washers
- [x] 32x M2x10mm
- [x] 8x M3x10mm
- [x] 8x M3x6mm
- [x] 12x M3x8mm
- [x] 1x Power Distribution Board - HOBBY KING LITE
- [x] 4x ESC - Favourite Little Bee 20A 2-4S
- [x] 1x Erle Robotics PXFmini
- [x] 1x Erle Robotics PXFmini Power Module
- [x] 1x Turnigy Graphene 1300mAh 4S 45C LiPo Pack w/ XT60
- [x] 4x EMAX RS2205 Brushless Motor
- [x] 4x GEMFAN 5045 GRP 3-BLADE Propellers
- [x] 4x 100mm Carbon Fiber Tube w/ Diameter of 12mm
- [x] 1x Pi Camera at 5MP

## Drone Parts
Using a modified version of the 3D printed Firefly drone (http://firefly1504.com)
### Mainframe
Files located in ```Firefly Drone Parts\MainFrame```
Print Bumper_v2_f.stl and 2x side.stld with 50% infill
Print Lower_plate_V2.stl, Top_plate_front_3mm.stl, and Top_plate_rear_3mm.stl with 25% infill
### Arms
Files located in ```Firefly Drone Parts\Arms```
Print 8x cliplock and rest of parts with 25% infill
### Cutting Jigs
Files located in ```Firefly Drone Parts\Drill Jig```
Print all parts with 25% infill
### Preparing Arms
Using the jigs, cut 500mm tube into 4x 100mm tubes
Drill 3mm and 4mm holes into the tubes through the jigs

## Assembly Log
![Nov. 1st Update 2](Logs/Nov1_2.jpg?raw=true "Nov. 1st")
___Nov. 1st Update 2___
Progress Update!

![Nov. 1st Update 1](Logs/Nov1_1.jpg?raw=true "Nov. 1st")
___Nov. 1st Update 1___
Soldered misconnection between PXFMini and Pi0 and wires between motors and ESCs

![Oct. 29th Update 1](Logs/Oct29.jpg?raw=true "Oct. 29th")
___Oct. 29th Update 1___
Mounted battery, Pi0, PXFMini, and Pi Camera.

![Oct. 24th Update 3](Logs/Oct24_5.jpg?raw=true "Oct. 24th")
![Oct. 24th Update 3](Logs/Oct24_6.jpg?raw=true "Oct. 24th")
___Oct. 24th Update 3___
Progress update!

![Oct. 24th Update 2](Logs/Oct24_2.jpg?raw=true "Oct. 24th")
![Oct. 24th Update 2](Logs/Oct24_3.jpg?raw=true "Oct. 24th")
![Oct. 24th Update 2](Logs/Oct24_4.jpg?raw=true "Oct. 24th")
___Oct. 24th Update 2___
Attaching motors to arms with the motor mounts with 2x M3 x 10mm and 4x M2 x 8mm

![Oct. 24th Update 1](Logs/Oct24_1.jpg?raw=true "Oct. 24th")
___Oct. 24th Update 1___
New parts arrived! Need to reorder propellers (2 are clockwise orientation) and a LiPo battery charger.

![Oct. 19th Update 2](Logs/Oct19_2.jpg?raw=true "Oct. 19th")
___Oct. 19th Update 2___
Began construction of actual drone! Still waiting for new motors to begin arm assembly but main frame is done!

![Oct. 19th Update 1](Logs/Oct19_1_1.jpg?raw=true "Oct. 19th")
![Oct. 19th Update 1](Logs/Oct19_1_2.jpg?raw=true "Oct. 19th")
___Oct. 19th Update 2___
PXFMini and power module arrived! I also went out and got various screws, nuts, and washers for the assembly.

![Oct. 18th Update 1](Logs/Oct18.jpg?raw=true "Oct. 18th")
___Oct. 18th Update 1___
Finished printing remaining pieces! Some issues cleaning support material off inner clamps but was sanded out. Carbon Fiber tube was cut with drill jig successfully!

![Oct. 16th Update 3](Logs/Oct16_3.jpg?raw=true "Oct. 16th")
___Oct. 16th Update 3___
Motors, Propellers, Carbon Fiber Tubes, Dampener Balls, and Camera arrived!

![Oct. 16th Update 2](Logs/Oct16_2.jpg?raw=true "Oct. 16th")
___Oct. 16th Update 2___
All Mainframe Materials printed successfully!

![Oct. 16th Update 1](Logs/Oct16_1.jpg?raw=true "Oct. 16th")
___Oct. 16th Update 1___
Printed Bumper_v2.stl and side.stl successfully! Lower_plate_V2 printed with wrong orientation and extra support material. Requeued.

## Resources

http://docs.erlerobotics.com/brains/pxfmini/software/apm
> Ardupilot (APM)

http://ksimek.github.io/2013/08/13/intrinsic/
> Explanation of camera matrices

https://homes.cs.washington.edu/~seitz/papers/cvpr97.pdf
> Paper detailing a proposal for coloured scene reconstruction by calibrated voxel colouring

https://people.csail.mit.edu/sparis/talks/Paris_06_3D_Reconstruction.pdf
> Presentation slide deck by Sylvian Paris on different methods for 3D reconstruction from multiple images

https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
> Stanford CS231A lecture notes on epipolar geometry and 3D reconstruction

http://pages.cs.wisc.edu/~chaol/cs766/
> Assignment page from the University of Wisconsin for uncalibrated stereo vision using epipolar geometry

http://journals.tubitak.gov.tr/elektrik/issues/elk-18-26-2/elk-26-2-11-1704-144.pdf
> Paper detailing a proposed 3D reconstruction pipeline given camera calibration matrices

https://rdmilligan.wordpress.com/2015/06/28/opencv-camera-calibration-and-pose-estimation-using-python/
> OpenCV Camera Calibration

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
> OpenCV Camera Calibration

https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
> Camera Calibration and 3D Reconstruction
