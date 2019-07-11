# circle
draw anything with complex fourier transformation
![github logo](https://github.com/biran0079/circle/blob/master/github.gif)

## Get Started
0 Install dependencies.
```
python3 -m pip install -r requirements
```
1 Find an image to draw.
![psyduck](https://github.com/biran0079/circle/blob/master/psyduck.png)
2 Find contour of the image. 
```
python3 find_contour.py psyduck.png
```
![find_contour](https://github.com/biran0079/circle/blob/master/find_contour.png)
3 Generate path for the contour.
Ideal number of samples in the path should be less than 500, otherwise your screen will be full of circles and animation will be slow to render.
```
python3 find_path.py psyduck.contour
```
4. Render animation.
```
python3 render.py psyduck.param
```
![find_path](https://github.com/biran0079/circle/blob/master/find_path.png)

![done](https://github.com/biran0079/circle/blob/master/psyduck.gif)
