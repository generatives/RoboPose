# RoboPose

This is a simple monocular positioning system for a small RaspberryPi based robot I was working on. It is not particularly effective or well implemented, just an idea I wanted to try.

The idea was to take a lot of photos from different known locations and angles. To locate the agent we can use OpenCV's "findEssentialMatrix" and "recoverPose" functions to find vectors pointing from the positions of the known images towards the current, unknown position. We then intersect the various vectors with one another and average the intersection points to find the current position.