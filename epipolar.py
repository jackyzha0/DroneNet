# 0 - Find Correspondence using the KLT feature point tracker.
# 1 - Determine the Fundamental Matrix, F.
# 2 - Recify images using F.
# 3 - Apply Birchfield and Tomasi calibated stereo correspondence algorithm to create a depth map.
# 4 - If the depth map generated is bad, determine new correspondence points by searching the space near the epipolar lines determined originally.
# 5 - Repeat from step 1, using the new correspondence points.
