author: YUEN Ho Shing
UID: 3035930943

All functions are successfully implemented in calin.py and epipolar.py.
And the code is self-explanatory with comments. 

What's worth noticing is that due to the difference in development environment, the paths of 
all input files (e.g. grid.cam) are represented by pathlib modules in my code. i.e. the 
default args of the parser in main() are all pathlib.Path objects in string form.
