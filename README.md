# circle
Draw anything with discrete fourier transform.
![github logo](https://github.com/biran0079/circle/blob/master/github.gif)

## Get Started
Install dependencies.
```
python3 -m pip install -r requirements
```

Find an image to draw.

![psyduck](https://github.com/biran0079/circle/blob/master/psyduck.png)


```
python3 find_contour.py psyduck.png
```

![find_contour](https://github.com/biran0079/circle/blob/master/find_contour.png)

```
python3 find_path.py psyduck.contour
```

![find_path](https://github.com/biran0079/circle/blob/master/find_path.png)

Ideal number of samples in the path should be less than 500, otherwise your screen will be full of circles and animation will be slow to render.


```
python3 render.py psyduck.param
```

![done](https://github.com/biran0079/circle/blob/master/psyduck.gif)
