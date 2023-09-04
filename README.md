## Image Processing and 3D Modeling with Python using Visual Hull

Firstly, this last week when you sent me chatGpt's - you were right, I had an error that was unrelated to the coordinate transforms and the physics was good as is, sorry for the slight lapse I was out of the zone for those 48 hours.
On to the README.

This Python script is designed for image processing and 3D modeling tasks. It uses video to capture a set of images, detect Aruco markers, extracts contours, performs camera calibration, and create a 3D reconstruction of the desired object.

## Prerequisites

Before running this script, make sure you have the following libraries installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- gmsh (`gmsh`)
- sys (`sys`)
- logging (`logging`)

You can install these libraries using `pip` or any other package manager.

## Usage

1. Clone this repository to your local machine or download the script.
2. Modify the script's configuration parameters according to your specific requirements, including folder paths, camera calibration parameters, and scaling factors.
3. Ensure that you have a set of images in the specified folder path, and the image filenames match the specified pattern.
4. Run the script using Python:

   ```bash
   python script.py
The script will process each image, detect Aruco markers, extract contours, perform camera calibration, and generate 3D models.
Configuration Parameters
folder_path: The path to the folder containing the input images.
imagename_pattern: The pattern for matching image filenames.
Camera calibration parameters (CMTX and DIST): These parameters define the camera's intrinsic matrix and distortion coefficients.
centerpoint: The coordinates of the centerpoint in the world coordinate system.
aruco_points: Dictionary defining Aruco marker points in the world coordinate system.
Other parameters related to image processing and modeling.
Output
The script generates 3D models based on the detected contours and Aruco markers. These models are written to Gmsh format files with the .msh extension.

Troubleshooting
If you encounter any issues or errors while running the script, please refer to the error messages and ensure that the required libraries are correctly installed.
License
This script is provided under the MIT License. Feel free to modify and use it for your projects.

## Results 3/9/23

Small hook that Joose measured, approximately 2mm error.  It is too large.  This could be due to several factors: the need to blur the image for best edge detection, an edge that is detected on the 'outside' edge of the hook blur (extra pixel of innacuracy), need to de-res the image for calibration, imperfect calibration due to the need to fix the focal length.  I do not think this method is worth continuing to research on its own.  It is possible that creating a 360 degree capture program with aruco cubes and open sourcing it would be novel and give amateur users like young students or tinkerers an alternative to polycam.  This is possible with the current setup and changes only one fundamental portion of the code.

Before blur
![SDcard_jpg_unprocessed](https://github.com/nichfi/NOZ_hook/assets/129064580/86f59cad-234e-4037-a99c-05999ba96375)

After blur
![SDcard_jpg_downscaled](https://github.com/nichfi/NOZ_hook/assets/129064580/ece8c396-bf98-4d7f-9a2d-bcc2fcc51d15)


I have cut some cubes on Sunday and printed out 12 aruco markers to use on them.  I think this would be worth a shot, but in the past you have stated that this hook issue only should use the one sided view so I did not focus on it.  This has been somewhat confusing, as visual hull is most effective for 3D measurements that require a complete object, in this case I believe that using diplib (https://pypi.org/project/diplib) would be a more worthy solution.  Additionally, if the typical caliper measurement were to be displaced by a phone application, I would expect that it offers a substantial improvement to the traditional methods.  Both polycam and any homebrewed/open-source software wll take more time than caliper measurement.  The only digital application I can see would be if the crane was in a precarious location.  For that, a 2D method could likely be utilized by controlling the exact location of the crane and camera in an area where humans could not reach.

To summarize:  For the konecranes project, I think another method would work better, but this does have the opportunity to be a decent solution for those looking to explore the properties of visual hull.

I have a few solutions to the inaccuracy, we can talk about them tomorrow.

ex. Dynamic camera matrices, aruco coordinate and contour detecting on different resolution images

Measurement Results: 
https://docs.google.com/document/d/1WT4fFXc_9t_Cw0HptHW2BK2zd4y0bLuXEGE4vyNoMW8 


