# Hand Brush 🖌️
Machine learning paint program that uses your hand as a brush!

<break>

`Note: Demos (gifs) will take a bit of time to load coz they are heavy 🧸`

<brek>


###  Demo of the program!
---
For hand traking my application here is using the pretrained ``SSD with MobilenetV1` model from [EvilPort2](https://github.com/EvilPort2)'s
hand tracking [repository](https://github.com/EvilPort2/Hand-Tracking). The paint toolbox uses the code from [acl21](https://github.com/acl21)'s Webcam Paint OpenCV [repository](https://github.com/acl21/Webcam_Paint_OpenCV). Hand Brush program is truly a combination of those two repositories and i highly recommend you to check out their repositories 🦔

<break>

![paint program demo][paint]

<break>


### Hand detection demo!
---

This demo uses [EvilPort2](https://github.com/EvilPort2)'s
hand tracking [model](https://github.com/EvilPort2/Hand-Tracking) but you can also try the [victordibia](https://github.com/victordibia)'s hand tracking [model](https://github.com/victordibia/handtracking). I tried both and got better result from the first one.

<break>

![hand detect demo][detect]

<break>


### Hand pointer demo!
---
This demo uses the [akshaybahadur21](https://github.com/akshaybahadur21)'s Hand Movement Tracking [code](https://github.com/akshaybahadur21/HandMovementTracking) for creating the pointer trail.

<break>

![pointer demo][pointer]

<break>

### Contour demo!
---
This one is just for fun and learning [How can I find contours inside ROI using opencv and Python?](https://stackoverflow.com/questions/42004652/how-can-i-find-contours-inside-roi-using-opencv-and-python)

<break>

![pointer demo][contour]

<break>

### Jupyter Notebooks
---
If you want to learn how does it work internally, the best way is to follow the Jupyter Notebooks i have included in this repository.

| Notebook | Purpose
| --- | --- |
|[detecthand.ipynb](./detecthand.ipynb)| This notebook will give you the base code for all the other notebooks and for the application itself.|
| [gethand.ipynb](./gethand.ipynb)| Here you'll learn to extract the detected hand as separate image data.|
|[centeroid.ipynb](./centeroid.ipynb)| We will calculate the centeroid point in this notebook.|
|[countour.ipynb](./contour.ipynb)| We will find the contours in the detected hand by taking it as ROI (Region Of Interest)|

![Jupyter Notebooks][notebooks]



<break>

`If you find any issue in the code OR want any help please create an issue in the issue tracker :)`

<break>

## References
1. https://github.com/EvilPort2/Hand-Tracking
2. https://github.com/acl21/Webcam_Paint_OpenCV
3. https://github.com/akshaybahadur21/HandMovementTracking
4. https://github.com/victordibia/handtracking




[detect]: ./assets/detect.gif "hand detect demo"
[paint]: ./assets/paint.gif "paint program demo"
[contour]: ./assets/contour.gif "contour demo"
[pointer]: ./assets/pointer.gif "pointer demo"
[notebooks]: ./assets/notebooks.png "contour demo"
