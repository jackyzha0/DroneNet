# DroneNet
## Jacky Zhao
### Science Fair Project 2018-19

## Summary
Using neural networks controlled drone swarms for optimized 3D reconstruction

## Drone Assembly
Using the 3D printed Firefly drone (http://firefly1504.com)
### Mainframe
Files located in ```Firefly Drone Parts\MainFrame```
Print Bumper_v2_f.stl and 2x side.stld with 50% infill
Print Lower_plate_V2.stl, Top_plate_front_3mm.stl, and Top_plate_rear_3mm.stl with 25% infill

## TODO
- [ ] Decide on 3D construction methodology
- [ ] Cost function for measuring accuracy of 3D reconstruction
- [x] Print Bumpers (Completed Oct. 16th)
- [ ] Print Sides
- [ ] Print Lower Plate
- [ ] Print Top Plate
- [ ] Print Arms
- [ ] Print Carbon Fiber Jigs
- [ ] Assemble Drones
- [x] Order PXFMini parts (Completed Oct. 15th)
- [x] Order Firefly Parts (Completed Oct. 12th)
- [x] Realtime rendering of 3D Space (Completed Oct. 2)
- [x] Particle based display for drone (Completed Sept. 28)
- [x] Create initial variables for drones (Completed Sept. 26)
- [x] Create drone class (Completed Sept. 26)

## Materials
- [x] Erle Robotics PXFmini
- [x] Erle Robotics PXFmini Power Module
- [x] 4x Gemfan 2040 3-Blade Propellers 2.0 Inch Triblade
- [x] 4x FPVDrone 1104 7500KV Brushless Motor
- [x] 4x 100mm Carbon Fiber Tube w/ Diameter of 12mm
- [ ] 4x M3 Washers
- [ ] 32x M2x10mm
- [ ] 8x M3x10mm
- [ ] 8x M3x6mm
- [ ] 12x M3x8mm
- [ ] 4x M3 Locknuts
- [ ] 2x M3x22mm spacers
- [x] 4x Dampener Balls
- [ ] 4x M3x32mm
- [x] 1x Pi Camera at 5MP

## 3D Printing Log
![Oct. 16th](Logs/Oct16.jpg?raw=true "Oct. 16th")
Printed Bumper_v2.stl and side.stl successfully! Lower_plate_V2 printed with wrong orientation and extra support material. Requeued.

## Resources
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
> Misc.

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
> Misc.

https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
> Misc.
