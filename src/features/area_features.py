from scipy.spatial import ConvexHull
import numpy as np

def create_polygon_features(points):
    '''Takes as an input a n x 2 dimensional numpy array with two columns as x and y
    coordinates and 'n' being number of points.
    For example : points = np.array([[0,0],[0,1],[1,0],[1,1])
    Returns the area, perimeter and a few other features after calculating
    the convex hull from that set of points
    '''
    # Generate a convex hull, which is a set of boundary points that enclose all other points
    hull = ConvexHull(points)
    hull_index = hull.vertices
    
    # Initialize the array which stores the convex hull
    hull_pts = np.zeros([len(hull_index),2]) 
    for i in range(len(hull_index)):
        hull_pts[i,:] = points[hull_index[i]]
     
    # Feature 1
    # Calculate area of the polygon defined by the convex hull
    # Sourced from https://stackoverflow.com/questions/19873596/convex-hull-area-in-python
    lines = np.hstack([hull_pts,np.roll(hull_pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    
    # Feature 2
    # Calculate perimeter of the points defined by the convex hull
    perimeter = 0
    for simplex in hull.simplices:
        perimeter = perimeter + np.linalg.norm(points[simplex, 0] - points[simplex, 1])
    
    # Feature 3
    # Total distance travelled by the curve in the x-direction
    x_path = 0
    for simplex in hull.simplices:
        x_path = x_path + abs((points[simplex, 0] - points[simplex, 1])[0])
    
    # Feature 4
    # Total distance travelled by the curve in the y-direction 
    y_path = 0
    for simplex in hull.simplices:
        y_path = y_path + abs((points[simplex, 0] - points[simplex, 1])[1])
    
    # Feature 5  
    # An indirect measure in angles that determines how much of the curve is along x vs y direction.
    path_phase = np.arctan(y_path/x_path) * 57.2
    
    return area, perimeter, x_path, y_path, path_phase
