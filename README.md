# DroneNet
### Jacky Zhao - Science Fair Project 2018-19

## Summary
Using neural networks controlled drone swarms for optimized 3D reconstruction.

## TODO
- [ ] Install new propellers
- [ ] Fix up arm wiring
- [ ] Write function to calibrate camera intrinsic matrix
- [ ] Get location from PXFMini
- [ ] Build and compile APM for PXFMini and Pi0
- [ ] Evaluate types of 3D reconstruction
- [ ] Evaluate types of machine learning
- [ ] Cost function for measuring accuracy of 3D reconstruction
- [ ] Construct Model
- [ ] Drone Test Flight
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
Using the 3D printed Firefly drone (http://firefly1504.com)
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
