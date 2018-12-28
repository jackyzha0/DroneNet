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
    - YOLOv1 implemented with SqueezeNet, input size [375, 375]
    - Dataset
      - [Deprecated] Kitware - VIRAT Dataset
      - KITTI Vision Benchmark - Bird's Eye View
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
- [ ] Check labels for accuracy
- [ ] Interpret Output
- [ ] imshow Output
- [ ] Calculate IOU and mAP

- [ ] Look into SparkFun nRF52832 Breakout
- [ ] Build and compile APM for PXFMini and Pi0
- [ ] Get location from PXFMini
- [ ] Draw on MapDisp
- [ ] Evaluate types of 3D reconstruction
- [ ] Evaluate types of machine learning
- [ ] Create training scenario
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

## detectionNet Explained
A modified version of YOLOv1 with SqueezeNet is implemented

Dimensionality of the next layer can be computed as follows:
![fig. 1](http://mathurl.com/ybw6b7yt.png)

| Layer | Output Dimensions | Filter Size | Stride | Depth |
|-------|------------|-------------|--------|-------|
| images | [375, 375, 3] | - | - | - |
| conv1 | [188, 188, 96] | [7, 7] | [2, 2] | 96 |
| maxpool1 | [93, 93, 96] | [3, 3] | [2, 2] | - |
| fire1 | [93, 93, 128] | - | - | 128 |
| fire2 | [93, 93, 128] | - | - | 128 |
| fire3 | [93, 93, 256] | - | - | 256 |
| maxpool2 | [46, 46, 256] | [3, 3] | [2, 2] | - |
| fire4 | [46, 46, 256] | - | - | 256 |
| fire5 | [46, 46, 384] | - | - | 384 |
| fire6 | [46, 46, 384] | - | - | 384 |
| fire7 | [46, 46, 512] | - | - | 512 |
| maxpool3 | [23, 23, 512] | [3, 3] | [2, 2] | - |
| fire8 | [23, 23, 512] | - | - | 512 |
| maxpool4 | [5, 5, 512] | [7, 7] | [4, 4] | - |
| conv2 | [5, 5, 27] | [1, 1] | [1, 1] | 27 |

Input Format:
img - [batchsize, 375, 375, 3]

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

## Data formatting of the KITTI Vision Dataset
### Images
Of format 000000.png <br/>
Training set size: ~7500 images
Testing set size: ~7500 images
Various dimensions from [1224, 370] to [1242, 375]
##### Processing
1. Resizing all uniform [1242, 375] with cubic interpolation
2. Crop random to [375, 375]
3. Normalize to range [0., 1.]

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

## Erle Robotics PXFMini SSH
Connect to WiFi network "erle-robotics-frambuesa"
ssh erle@10.0.0.1

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

https://www.unmannedtechshop.co.uk/tattu-1550mah-14-8v-45c-4s1p-lipo-battery-pack/
> Battery Information

https://hobbyking.com/en_us/favourite-little-bee-20a-2-4s-esc-no-bec.html
> ESC Information

https://hobbyking.com/en_us/hobby-king-quadcopter-power-distribution-board-lite.html
> Power Distribution Board

https://emax-usa.com/emax-rs2205-racespec-motor-cooling-series.html
> Motor

https://www.banggood.com/4PCS-Bullnose-5045-2-Blade-Propellers-2CW2CCW-For-250-280-310-Frame-Kits-p-985839.html?ID=224&cur_warehouse=CN
> Propeller

https://www.dronetrest.com/t/what-to-consider-when-buying-a-esc-for-your-multirotor/1305
> Determining Correct ESC

https://www.dronetrest.com/t/how-to-choose-the-right-motor-for-your-multicopter-drone/568
> Motor Information
