import logging
import re

import numpy as np

from simulation_camera import SimulationCamera


class Image():
    """

    Array with associated information.

    ...
    Parameters
    ----------
    filename : str
        Name of image file.

    Attributes
    ----------
    im : np.array
        Image.

    """

    def __init__(self, filename: str):
        self.filename = filename

        # For some reason taking square root improves estimating transform.
        self.im = np.sqrt(self.load())

    def load(self) -> np.array:
        """Load image from file.

        Raises
        ------
        NotImplementedError
            if not implemented by a subclass.

        Returns
        -------
        im : np.array
            Array representing the image.

        """
        raise NotImplementedError


class LuminanceCameraImage(Image):
    '''Array representing measurement image.'''

    def load(self):
        # Overwrites the superclass method.
        return self._load_pf()

    def _load_pf(self):
        # Thanks, Gregor!
        with open(self.filename, mode='rb') as f:

            f.readline()

            m = re.search(rb'(?<=Lines=)\d+', f.readline())
            nrows = int(m.group())

            m = re.search(rb'(?<=Columns=)\d+', f.readline())
            ncols = int(m.group())

            f.seek(-4 * nrows * ncols, 2)

            im = np.fromfile(f, dtype=np.float32).reshape((nrows, ncols))

            logging.debug('\tLoaded luminance camera image.\n')

        return im


class SimulationImage(Image):
    """

    Array representing simulation image.

    ...

    Parameters
    ----------
    camera_filename : str, optional
        Name of file with camera settings (.camera).

    Attributes
    ----------
    camera : SimulationCamera
        Camera settings.

    """

    def __init__(self, image_filename, camera_filename=None):
        super().__init__(image_filename)

        self.camera_filename = camera_filename
        self.camera = None
        if camera_filename:
            self.camera = SimulationCamera(camera_filename)

    def load(self):
        # Overwrites the superclass method.
        return self._load_txt()

    def _load_txt(self):

        with open(self.filename, mode='r') as f:

            for _ in range(3):
                f.readline()

            im = np.loadtxt(f, dtype=np.float32)[::-1]

            logging.debug('\tLoaded simulation image.\n')

        return im

    def add_camera(self, camera_filename: str):
        """Add camera to image.

        Parameters
        ----------
        camera_filename : str
            Name of file with camera settings (.camera).

        """
        self.camera_filename = camera_filename
        self.camera = SimulationCamera(camera_filename)
        return None

    def remove_camera(self):
        """Remove camera from image.

        """
        self.camera_filename = None
        self.camera = None
        return None

    def new_camera(self, Mc) -> SimulationCamera:
        """Compute new camera settings.

        Parameters
        ----------
        Mc : array_like
            2x3 transformation matrix comprising uniform scaling, rotation and
            translation, with origin in the center of the image.

        Returns
        -------
        camera : SimulationCamera
            New camera settings.

        """
        camera = self.camera.new_camera(Mc, len(self.im))
        return camera
