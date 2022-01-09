import logging

import numpy as np


class SimulationCamera():
    """

    Simulation camera settings.

    ...

    Parameters
    ----------
    filename : str, optional
        Name of camera file (.camera).
    method : {'r', 'w'}
        Read settings from file if 'r', write settings to file if 'w',
        default 'r'.
    pos : tuple of floats, optional
        Position of camera in optical axis system, in mm.
    backdir : tuple of floats, optional
        Unit vector specifying negative viewing direction.
    rightdir : tuple of floats, optional
        Unit vector specifying direction in the viewing plane,
        perpendicular to the viewing direction (clockwise).
    angle : float, optional
        Vertical viewing angle in radians.

    """

    def __init__(self, filename, method='r',
                 pos=None, backdir=None, rightdir=None, angle=None):
        self.filename = filename
        if method == 'r':
            pos, backdir, rightdir, angle = self._read_params()
        self.pos = pos
        self.backdir = backdir
        self.rightdir = rightdir
        self.angle = angle

    def __repr__(self):
        s = self.filename + '\n'
        s += '\tposition: {} mm\n'.format(np.round(self.pos, 2))
        updir = np.cross(self.backdir, self.rightdir)
        s += '\tup direction: {}\n'.format(np.round(updir, 6))
        s += '\tview angle: {:.2f}'.format(self.angle / np.pi * 180) + ' deg\n'
        return s

    def _read_params(self):
        """Read camera parameters.

        Returns
        -------
        pos : array of floats, optional
            Position of camera in optical axis system, in mm.
        backdir : array of floats, optional
            Unit vector specifying negative viewing direction.
        rightdir : array of floats, optional
            Unit vector specifying direction in the viewing plane,
            perpendicular to the viewing direction (clockwise).
        angle : float, optional
            Vertical viewing angle in radians.

        """

        with open(self.filename, mode='r') as f:
            header = f.readline()

        header = str(header).split(sep=',')

        def read_triplet(s):
            x = float(header[s].split(sep='(')[-1])
            y = float(header[s+1])
            z = float(header[s+2].split(sep=')')[0])
            return np.asarray([x, y, z])

        pos = read_triplet(2)  # Point(x, y, z)
        backdir = read_triplet(5)  # 1st UnitVector(dx, dy, dz)
        rightdir = read_triplet(8)  # 2nd UnitVector(dx, dy, dz)

        angle = float(header[12])  # 2nd float after rightdir

        logging.debug('\tRead camera parameters.\n')

        return pos, backdir, rightdir, angle

    def new_camera(self, Mc, image_height):
        """Compute new camera parameters.

        Parameters
        ----------
        Mc : array_like
            2x3 transformation matrix comprising uniform scaling, rotation and
            translation, with origin in the center of the image.
        image_height : int
            Image height in pixels.

        Returns
        -------
        camera : SimulationCamera
            New camera settings.

        """

        # new viewing angle
        scale_factor = (np.sum(Mc[:, 0] ** 2)) ** 0.5
        new_angle = 2 * np.arctan(1 / scale_factor * np.tan(self.angle / 2))

        # rotation angle
        phi = 1 / scale_factor * np.arcsin(Mc[1, 0])

        # old up direction
        updir = np.cross(self.backdir, self.rightdir)

        # new right direction
        new_rightdir = -np.sin(phi) * updir + np.cos(phi) * self.rightdir

        # new up direction
        new_updir = np.cross(self.backdir, new_rightdir)

        # new position
        d = (np.sum(self.pos ** 2)) ** 0.5
        pixel = 2 * d * np.tan(new_angle / 2) / image_height  # 1 pixel in mm
        new_pos = self.pos -\
            Mc[0, 2] * pixel * new_rightdir +\
            Mc[1, 2] * pixel * new_updir

        logging.debug('\tCalculated new camera parameters.\n')

        camera = SimulationCamera(self.filename,
                                  method='w',
                                  pos=new_pos,
                                  backdir=self.backdir,
                                  rightdir=new_rightdir,
                                  angle=new_angle)
        return camera

    def save_camera(self):
        """Save camera parameters.

        """

        # read the first settings in the file
        with open(self.filename, mode='r') as f:
            ref_header = f.readline()

        ref_header = str(ref_header).split(sep=',')

        header = []
        header = ref_header[:1]

        # camera position
        header.append('Axis(Point({:.2f}'.format(self.pos[0]))
        header.append('{:.2f}'.format(self.pos[1]))
        header.append('{:.2f})'.format(self.pos[2]))

        # read unit vector
        def aux(triplet):
            header.append('UnitVector({:.7f}'.format(triplet[0]))
            header.append('{:.7f}'.format(triplet[1]))
            header.append('{:.7f}))'.format(triplet[2]))
            return None

        aux(self.backdir)
        aux(self.rightdir)

        header.append(ref_header[11])

        # viewing angle
        header.append('{:.7f}'.format(self.angle))

        header.extend(ref_header[13:])

        header = ','.join(header)

        with open(self.filename, mode='a') as f:
            f.write('\n' + header)

        logging.debug('\tCamera saved.\n')

        return None

    def project_aligned_camera(self, aligned_camera_filename):
        """Project position of aligned camera to plane that is normal to
        unaligned camera's negative viewing direction and passes through its
        position.

        Parameters
        ----------
        aligned_camera_filename : str
            Name of file with camera settings (.camera).

        """

        aligned_camera = SimulationCamera(aligned_camera_filename)

        # negative viewing direction should be the same
        if not np.allclose(self.backdir, aligned_camera.backdir, atol=1e-2):
            raise ValueError

        offset = aligned_camera.pos - self.pos
        offset_in_plane = offset -\
            np.dot(offset, self.backdir) * self.backdir

        aligned_camera.pos = self.pos + offset_in_plane

        logging.info('\tComputed new aligned camera position.\n')

        aligned_camera.save_camera()

        return None
