import logging

import cv2
import numpy as np
from scipy.signal.windows import tukey


class Align():
    """

    Aligning simulation camera image to luminance camera as in paper by
    B. S. Reddy and B. N. Chatterji, "An FFT-based technique for
    translation, rotation, and scale-invariant image registration," in
    IEEE Transactions on Image Processing, vol. 5, no. 8, pp. 1266-1271,
    Aug. 1996, doi: 10.1109/83.506761.

    ...

    Parameters
    ----------
    sim : array_like
        Simulation camera image.
    meas : array_like
        Luminance camera image.

    Attributes
    ----------
    h : int
        Height of simulation camera image.
    w : int
        Width of simulation camera image.
    mfsim : array_like
        Fourier log-magnitude spectrum of sim, of dimension w x w.
    mfsim_lp : array_like
        Log-polar transform of mfsim, of dimension w x w.
    meas : array_like
        Luminance camera image, resized to the size of simulation camera image.
    mfmeas : array_like
        Fourier log-magnitude spectrum of meas, of dimension w x w.
    mfmeas_lp : array_like
        Log-polar transform of mfmeas, of dimension w x w.
    Mc : array_like
        2x3 transformation matrix comprising uniform scaling, rotation and
        translation, with origin in the center of the image.
    Mu : array_like
        2x3 transformation matrix comprising uniform scaling, rotation and
        translation, with origin in the upper left corner of the image.
    new_sim : array_like
        Transformed simulation camera image.

    """

    def __init__(self, sim, meas):
        self.sim = sim
        self.h, self.w = sim.shape
        self.mfsim = None
        self.mfsim_lp = None
        self.meas = cv2.resize(meas, sim.shape[::-1])
        self.mfmeas = None
        self.mfmeas_lp = None
        self.Mc = None
        self.Mu = None
        self.new_sim = None

    def dft(self):
        """Compute 2D discrete Fourier log-magnitude spectra mfsim and mfmeas
        of sim and meas.

        """

        cx, cy = self.w // 2, self.h // 2

        def aux(im):

            # 2D discrete Fourier transform
            fim = np.fft.fft2(im)

            # log.magnitude spectrum
            mfim = np.log(np.abs(fim))

            # bring (0, 0) to the center of the image
            mfim = np.roll(mfim, cy, axis=0)
            mfim = np.roll(mfim, cx, axis=1)

            # the image should be square, otherwise we would have problems
            # estimationg rotation angle
            mfim = cv2.resize(mfim, (self.w, self.w))

            return mfim

        self.mfsim = aux(self.sim)
        self.mfmeas = aux(self.meas)

        return None

    def log_polar(self, alpha):
        """Transform mfsim and mfmeas to log-polar coordinates,
        mfsim_lp and mfmeas_lp.

        Parameters
        ----------
        alpha : float
            Tukey window function parameter.

        """

        cx = self.w // 2

        def aux(mfim):

            # log-polar transform
            mfim_lp = cv2.warpPolar(src=mfim, dsize=(self.w, self.w),
                                    center=(cx, cx), maxRadius=cx,
                                    flags=cv2.WARP_POLAR_LOG)

            # window function to reduce edge effects (instead of a high-pass
            # filter)
            window = tukey(self.w, alpha=alpha)

            return window * mfim_lp

        self.mfsim_lp = aux(self.mfsim)
        self.mfmeas_lp = aux(self.mfmeas)

        return None

    def scale_rotation(self):
        """Estimate scale factor and rotation angle, compute Mc and Mu.

        """

        cx = self.w // 2

        (sx, sy), response = cv2.phaseCorrelate(self.mfsim_lp, self.mfmeas_lp)

        scale = np.exp(-sx * np.log(cx) / self.w)
        phi = sy / self.w * 2 * np.pi

        logging.info(
            'Estimated scale and rotation:\n' +
            '\tresponse: {:.2f}\n'.format(response) +
            '\tscale factor: {:.4f} ({:.1f} pixels)\n'.format(scale, sx) +
            '\trotation angle: {:.2g} deg ({:.1f} pixels)\n'.format(phi / np.pi * 180, sy)
        )

        m11, m21 = np.cos(phi), np.sin(phi)
        self.Mc = scale * np.asarray([[m11, -m21, 0], [m21, m11, 0]])
        self.Mu = self._uncenter(self.Mc)

        return None

    def _uncenter(self, Mc):
        # Rewrite Mc as Mu (move coordinate system origin from center of image
        # to upper left corner.

        h, w = self.h, self.w
        x_shift = - 0.5 * (Mc[0, 0] * w + Mc[0, 1] * h) + Mc[0, 2] + w / 2
        y_shift = - 0.5 * (Mc[1, 0] * w + Mc[1, 1] * h) + Mc[1, 2] + h / 2

        Mu = np.copy(Mc)
        Mu[:, 2] = [x_shift, y_shift]

        return Mu

    def _center(self, Mu):
        # Rewrite Mu as Mc (move coordinate system origin from upper left
        # corner to center of image.

        h, w = self.h, self.w
        x_shift = 0.5 * (Mu[0, 0] * w + Mu[0, 1] * h) + Mu[0, 2] - w / 2
        y_shift = 0.5 * (Mu[1, 0] * w + Mu[1, 1] * h) + Mu[1, 2] - h / 2

        Mc = np.copy(Mu)
        Mc[:, 2] = [x_shift, y_shift]

        return Mc

    def translation(self):
        """Estimate translation, compute Mc and Mc.

        """

        # scale and rotate simulation image
        sim_sr = cv2.warpAffine(src=self.sim, M=self.Mu,
                                dsize=(self.w, self.h))

        # find translation
        (sx, sy), response = cv2.phaseCorrelate(sim_sr, self.meas)

        self.Mu[:, 2] += [sx, sy]
        self.new_sim = cv2.warpAffine(src=self.sim, M=self.Mu,
                                     dsize=(self.w, self.h))
        self.Mc = self._center(self.Mu)

        logging.info(
            'Estimated translation:\n' +
            '\tresponse: {:.2f}\n'.format(response) +
            '\tx translation: {:.1f} pixels\n'.format(self.Mc[0, 2]) +
            '\ty translation: {:.1f} pixels\n'.format(self.Mc[1, 2])
        )

        return None

    def find_M(self, alpha=0.5):
        """Estimate transformation matrix.

        Parameters:
        -----------
        alpha: float, optional
            Tukey window function parameter, default 0.5.

        Returns:
        --------
        Mc: array_like
            2x3 transformation matrix comprising uniform scaling, rotation and
            translation, with origin in the center of the image.

        """

        self.dft()
        self.log_polar(alpha)
        self.scale_rotation()
        self.translation()

        return self.Mc
