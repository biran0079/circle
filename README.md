# circle
Draw anything with discrete fourier transform.
![github logo](https://user-images.githubusercontent.com/661451/61032364-27c1ee00-a376-11e9-9535-443d3fc5af2c.gif)

## Get Started
Install dependencies.
```
python3 -m pip install -r requirements
```

Find an image to draw.

![psyduck](https://user-images.githubusercontent.com/661451/61032306-095bf280-a376-11e9-968e-35e9cfe8a427.png)


```
python3 find_contour.py psyduck.png
```

![find_contour](https://user-images.githubusercontent.com/661451/61032397-37d9cd80-a376-11e9-8f26-a154da89c034.png)

```
python3 find_path.py psyduck.contour
```

![find_path](https://user-images.githubusercontent.com/661451/61032416-43c58f80-a376-11e9-9888-74c476ed0003.png)

Ideal number of samples in the path should be less than 500, otherwise your screen will be full of circles and animation will be slow to render.


```
python3 render.py psyduck.param
```

![psyduck](https://user-images.githubusercontent.com/661451/61032543-7c656900-a376-11e9-85be-af76735843b5.gif)

You can literally draw anything!

![doraemon](https://user-images.githubusercontent.com/661451/61033903-33fb7a80-a379-11e9-8b1f-f57799977bdc.gif)
