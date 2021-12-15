import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys


# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# Generate data
# pos = np.random.normal(size=(100000, 3), scale=0.2)
# centers = np.random.normal(size=(50, 3))
# indexes = np.random.normal(size=100000, loc=centers.shape[0]/2., scale=centers.shape[0]/3.)
# indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
# scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
# pos *= scales
# pos += centers[indexes]
# print(pos.shape)
# print(pos)
V = np.load('../data/data.npy')
p = []
for i in range(len(V)):
    p.append(V[i][0])
pos = np.array(p)

scatter = visuals.Markers()
scatter.set_data(pos, edge_color=(1, 1, 1, 1), face_color=(0, 1, 1, 1), size=5)

view.add(scatter)
view.camera = 'turntable'

# Add a colored 3D axis for orientation
# axis = visuals.XYZAxis(parent=view.scene)


if __name__ == '__main__':
    vispy.app.run()

