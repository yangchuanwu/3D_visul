import os
import numpy as np

from tqdm import tqdm
from skimage import io
from vispy import app, gloo
from vispy.gloo import Program
from vispy.gloo.util import _screenshot
from vispy.util.transforms import perspective, translate, rotate
from moviepy.editor import VideoClip

DATA_ROOT = '../data/1A-DB/1A-2dtiff/'
SAVE_ROOT = '../result1/1A-DB/1A-2dtiff/'
INFO_NAME = 'C1_.info'
W, H = 800, 600
V = np.load('../data/data_256.npy')

vert = """
    // Uniforms
    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;   
     
    uniform float u_size;
    uniform float u_clock;

    // Attributes
    attribute vec3 a_position;
    attribute vec4 a_color;

    // Varying
    varying vec4 v_color;

    void main(void)
    {
        v_color = a_color;    
        gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    }
"""

frag = """
    // varying
    varying vec4 v_color;

    void main()
    {
        gl_FragColor = v_color;
    }
"""


# ---------------------------------------------------------------------------------------------------
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(W, H))

        # Build program
        self.program = Program(vert, frag)
        self.program.bind(gloo.VertexBuffer(V))

        # Set uniforms and attributes
        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 2.0, 10.0)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection

        gloo.set_state('additive', depth_test=False)
        self.program['u_clock'] = 0.0

    # -------------------------------------------------------------------
    def updata_matrix(self):
        self.model = np.dot(rotate(self.theta, (0, 1, 0)),
                            rotate(self.phi, (0, 0, 1)))
        self.program['u_model'] = self.model
        self.update()

    # -------------------------------------------------------------------
    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])

        self.projection = perspective(45.0, self.size[0] /
                                      float(self.size[1]), 1.0, 1000.0)
        self.program['u_projection'] = self.projection

    # -------------------------------------------------------------------
    def animation(self, t):
        """ Added for animation with MoviePy """
        self.theta, self.phi = 5 * t, 0.0
        self.updata_matrix()
        gloo.clear('black')
        self.program.draw('points')
        return _screenshot((0, 0, self.size[0], self.size[1]))[:, :, :3]


if __name__ == '__main__':
    canvas = Canvas()
    canvas.show()
    clip = VideoClip(canvas.animation, duration=100)
    clip.write_videofile('rotate_cube.mp4', fps=20)
