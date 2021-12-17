import os
import numpy as np

from skimage import io
from vispy import app, gloo
from vispy.gloo import Program
from vispy.gloo.util import _screenshot
from vispy.util.transforms import perspective, translate, rotate
from moviepy.editor import VideoClip

DATA_ROOT = 'G:/Data for huige-211122/1A-DB/1A-2dtiff/'
SAVE_ROOT = 'G:/Data for huige-211122/1A-DB/'
INFO_NAME = 'C3_.info'
d, w, h = 697, 2015, 2048
THRESHOLD = 3e4
W, H = 800, 600

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


def read_data():
    """
    :return: imgs: [num * W * H]
    """
    data_root = DATA_ROOT
    save_root = SAVE_ROOT
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    info_name = INFO_NAME

    f = open(data_root + info_name)

    # # # -------------------------------------------
    # img1 = io.imread(data_root + 'C2_600.tif')
    # img2 = io.imread(data_root + 'C2_602.tif')
    # img = (img1 + img2) / 2
    # io.imsave(data_root + 'C2_601.tif', img)
    # exit()
    # # # -------------------------------------------

    print('Loading images...')
    v = []
    num = 0
    vtype = [('a_position', np.float32, 3),
             ('a_color', np.float32, 4)]
    # c = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
    #      [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
    for line in f:
        if '.tif' in line[1:11]:
            # print(line[1:11])
            img_file = data_root + line[1:11]
            img = io.imread(img_file)

            index = np.argwhere(img > THRESHOLD)
            for i in range(len(index)):
                pos = (np.hstack(([num], index[i])) / [d, w, h] * 2 - 1.).tolist()
                color = [0, img[index[i, 0], index[i, 1]]/65535, 0, 1]

                v.append((pos, color))
            num += 1

    print('Images load completely!!! ')
    V = np.array(v, dtype=vtype)
    # np.save(SAVE_ROOT + 'data_C3.npy', V)
    # print('Vertices construct successfully')
    return V


# ---------------------------------------------------------------------------------------------------
class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(W, H))
        # position and color
        V = read_data()

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
                            rotate(self.phi, (1, 0, 0)))
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
        self.theta, self.phi = 5 * t, 5 * t
        self.updata_matrix()
        gloo.clear('black')
        self.program.draw('points')
        return _screenshot((0, 0, self.size[0], self.size[1]))[:, :, :3]


if __name__ == '__main__':
    canvas = Canvas()
    canvas.show()
    clip = VideoClip(canvas.animation, duration=100)
    clip.write_videofile('rotate_C3.mp4', fps=20)
