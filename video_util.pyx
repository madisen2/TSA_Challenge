
cimport numpy as np

import numpy as np
from skvideo.io import FFmpegWriter

class VideoHelper(object):
    """This class wraps skvideo (from package sk-video) to help make videos
    using FFmpeg.

    Usage:

        with VideoHelper("output.mp4", fps=30, frameSize=(width, height)) as vh:
            for i in range(num_frames):
                with vh.frame() as buffer:
                    frameData = get_image_for_frame(i)
                    buffer.blit_flat_uint8[:, :] = frameData
    """

    def __init__(self, filename, frameSize, fps=30, compression=0.25):
        """

        Args:
            filename (str): Path to output mp4 file.

            frameSize (tuple): ``(width, height)``.

            fps (float): Frames per second of output video.

            compression (float): Approximate desired savings over lossless
                    encoding.  ``1`` is lossless.  Ranges from ``0.004`` to
                    ``1``.
        """
        nbuf = frameSize[0] * frameSize[1] * 3
        self._buf = np.zeros((frameSize[1], frameSize[0], 3), dtype=np.uint8)
        # Compression factor... roughly half per 6.
        if compression > 1 or compression < 0.004:
            raise ValueError("0.004 < compression < 1: compression {}".format(
                    compression))
        crf = int(round(6 * -np.log(compression) / np.log(2)))
        self._vid = FFmpegWriter(filename,
                inputdict={
                    '-r': str(fps),
                    '-s': '{}x{}'.format(*frameSize),
                },
                outputdict={
                    '-c:v': 'libx264',
                    '-preset': 'slow',
                    # If you want lossless, set crf to 0.  Each 6 halves the
                    # filesize, according to https://trac.ffmpeg.org/wiki/Encode/H.264#a1.ChooseaCRFvalue
                    '-crf': str(crf),
                },
        )


    def frame(self):
        self._buf[:, :, :] = 0.
        return _VideoFrame(self._vid, self._buf)


    def __enter__(self):
        return self


    def __exit__(self, typ, val, tb):
        self._vid.close()



cdef class _VideoFrame:
    cdef public np.uint8_t[:, :, ::1] _buf
    cdef public object _vid

    @property
    def blit_flat_float(self):
        """Used to copy a 1-d (flattened) RGB array of doubles into our buffer.

        Usage:
            frame.blit_flat_float[y_range, x_range] = buffer matching ranges
        """
        return _VideoFrameBlitFlatFloat(self)


    @property
    def blit_flat_uint8(self):
        """Used to copy a 1-d (flattened) RGB array of uint8 into our buffer.

        Usage:
            frame.blit_flat_uint8[y_range, x_range] = buffer matching ranges
        """
        return _VideoFrameBlitFlatUint8(self)


    @property
    def blit_flat_float_mono_as_alpha(self):
        """Used to copy a 1-d (flattened) array of doubles into our buffer,
        using a color to represent the monochromatic data.  Optionally, an
        alpha multiplier may be applied.

        Usage:
            frame.blit_flat_float_mono_as_alpha[y_range, x_range,
                    color, alpha_mult, alpha_pow] = src_array

        ``color``, ``alpha_mult``, and ``alpha_pow`` are optional.
        ``alpha_pow`` is the power (once scaled to (0, 1)) taken of the mono
        channel.
        """
        return _VideoFrameBlitFlatFloatMonoAlpha(self)


    @property
    def blit_flat_uint8_mono_as_alpha(self):
        """Used to copy a 1-d (flattened) array of uint8 into our buffer,
        using a color to represent the monochromatic data.  Optionally, an
        alpha multiplier may be applied.

        Usage:
            frame.blit_flat_uint8_mono_as_alpha[y_range, x_range,
                    color, alpha_mult, alpha_pow] = src_array

        ``color``, ``alpha_mult``, and ``alpha_pow`` are optional.
        ``alpha_pow`` is the power (once scaled to (0, 1)) taken of the mono
        channel.
        """
        return _VideoFrameBlitFlatUint8MonoAlpha(self)


    def __init__(self, vid, buf):
        self._vid = vid
        self._buf = buf


    def __enter__(self):
        return self


    def __exit__(self, typ, val, tb):
        self._vid.writeFrame(self._buf)



cdef class _VideoFrameBlitFlatFloat:
    cdef public _VideoFrame _frame
    def __init__(self, videoFrame):
        self._frame = videoFrame
    def __setitem__(self, key, item):
        if len(key) != 2:
            raise ValueError("Requires 2 keys")

        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef double[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[1], dst.shape[1])
        hs, he, ht = _read_slice(key[0], dst.shape[0])

        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                dst[h, w, 0] = max(0, min(255, int(src[i] * 255)))
                dst[h, w, 1] = max(0, min(255, int(src[i+1] * 255)))
                dst[h, w, 2] = max(0, min(255, int(src[i+2] * 255)))
                i += 3



cdef class _VideoFrameBlitFlatUint8:
    cdef public _VideoFrame _frame
    def __init__(self, videoFrame):
        self._frame = videoFrame
    def __setitem__(self, key, item):
        if len(key) != 2:
            raise ValueError("Requires 2 keys")

        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef np.uint8_t[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[1], dst.shape[1])
        hs, he, ht = _read_slice(key[0], dst.shape[0])

        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                dst[h, w, 0] = src[i]
                dst[h, w, 1] = src[i+1]
                dst[h, w, 2] = src[i+2]
                i += 3



cdef class _VideoFrameBlitFlatFloatMonoAlpha:
    cdef public _VideoFrame _frame
    def __init__(self, frame):
        self._frame = frame
    def __setitem__(self, key, item):
        cdef double amult = 1., apow = 1., u, uu
        cdef np.uint8_t[:] acolor = np.asarray([255, 255, 255], dtype=np.uint8)

        if len(key) < 2 or len(key) > 5:
            raise ValueError("len(key) must be between 2 and 5")
        if len(key) >= 5:
            apow = key[4]
        if len(key) >= 4:
            amult = key[3]
        if len(key) >= 3:
            acolor = np.asarray(key[2], dtype=np.uint8)

        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef double[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[1], dst.shape[1])
        hs, he, ht = _read_slice(key[0], dst.shape[0])
        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                u = max(0., min(1., src[i] ** apow * amult))
                uu = 1. - u
                dst[h, w, 0] = max(0, min(255, int(u * acolor[0]
                        + uu * dst[h, w, 0])))
                dst[h, w, 1] = max(0, min(255, int(u * acolor[1]
                        + uu * dst[h, w, 1])))
                dst[h, w, 2] = max(0, min(255, int(u * acolor[2]
                        + uu * dst[h, w, 2])))
                i += 1



cdef class _VideoFrameBlitFlatUint8MonoAlpha:
    cdef public _VideoFrame _frame
    def __init__(self, frame):
        self._frame = frame
    def __setitem__(self, key, item):
        cdef double amult8 = 1. / 255, amult = 1., apow = 1., u, uu
        cdef np.uint8_t[:] acolor = np.asarray([255, 255, 255], dtype=np.uint8)

        if len(key) < 2 or len(key) > 5:
            raise ValueError("len(key) must be between 2 and 5")
        if len(key) >= 5:
            apow = key[4]
        if len(key) >= 4:
            amult = key[3]
        if len(key) >= 3:
            acolor = np.asarray(key[2], dtype=np.uint8)

        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef np.uint8_t[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[1], dst.shape[1])
        hs, he, ht = _read_slice(key[0], dst.shape[0])
        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                u = max(0., min(1., (src[i] * amult8) ** apow * amult))
                uu = 1. - u
                dst[h, w, 0] = max(0, min(255, int(u * acolor[0]
                        + uu * dst[h, w, 0])))
                dst[h, w, 1] = max(0, min(255, int(u * acolor[1]
                        + uu * dst[h, w, 1])))
                dst[h, w, 2] = max(0, min(255, int(u * acolor[2]
                        + uu * dst[h, w, 2])))
                i += 1



cdef tuple _read_slice(key, int arrsize):
    cdef int s, e, t
    if isinstance(key, slice):
        if key.start is None:
            s = 0
        elif key.start < 0:
            s = arrsize + key.start
        else:
            s = key.start
        if key.stop is None:
            e = arrsize
        elif key.stop < 0:
            e = arrsize + key.stop
        else:
            e = key.stop
        t = key.step or 1
    elif isinstance(key, int):
        s = key
        e = key + 1
        t = 1
    else:
        raise ValueError("Not a slice or int! {}".format(key))
    return s, e, t

