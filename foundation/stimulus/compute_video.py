import io
import av
import numpy as np
from djutils import keys, rowproperty
from foundation.utils import video
from foundation.virtual.bridge import pipe_stim, pipe_gabor, pipe_dot, pipe_rdk
from foundation.virtual import utility, stimulus


# ---------------------------- Video ----------------------------

# -- Video Base --


class Video:
    """Video Stimulus"""

    @rowproperty
    def video(self):
        """
        Returns
        -------
        foundation.utils.video.Video
            video object
        """
        raise NotImplementedError()

    @rowproperty
    def directions(self):
        """
        Yields
        ------
        float
            direction -- degrees [0, 360)
        float
            onset of direction (seconds relative to first video frame)
        float
            offset of direction (seconds relative to first video frame)
        """
        raise NotImplementedError()


# -- Video Types --


@keys
class Clip(Video):
    """Clip Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.Clip,
        ]

    @rowproperty
    def video(self):
        clip = pipe_stim.Movie * pipe_stim.Movie.Clip * pipe_stim.Clip & self.key
        clip, start, end, fps = clip.fetch1("clip", "skip_time", "cut_after", "frame_rate")

        start, end = map(float, [start, end])
        start = round(start * fps)
        end = start + round(end * fps)

        frames = []
        reader = av.open(io.BytesIO(clip.tobytes()), mode="r")

        for i, frame in enumerate(reader.decode()):

            if i < start:
                continue
            if i == end:
                break

            frame = frame.to_image().convert(mode="L")
            frames.append(frame)

        return video.Video(frames, period=1 / fps)


@keys
class Monet2(Video):
    """Monet2 Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.Monet2,
        ]

    @rowproperty
    def video(self):
        frames, fps = (pipe_stim.Monet2 & self.key).fetch1("movie", "fps")
        frames = np.einsum("H W C T -> T H W C", frames)
        return video.Video.fromarray(frames, period=1 / float(fps))

    @rowproperty
    def directions(self):
        directions, onsets, duration, n_dirs, frac = (pipe_stim.Monet2 & self.key).fetch1(
            "directions", "onsets", "duration", "n_dirs", "ori_fraction", squeeze=True
        )
        assert len(directions) == len(onsets) == n_dirs

        # 0 degree is right, 90 is up
        directions = (90 - directions) % 360

        # direction offsets
        offsets = onsets + float(duration) / n_dirs * frac

        yield from zip(directions, onsets, offsets)


@keys
class Trippy(Video):
    """Trippy Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.Trippy,
        ]

    @rowproperty
    def video(self):
        frames, fps = (pipe_stim.Trippy & self.key).fetch1("movie", "fps")
        frames = np.einsum("H W T -> T H W", frames)
        return video.Video.fromarray(frames, period=1 / float(fps))


@keys
class GaborSequence(Video):
    """Gabor Sequence Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.GaborSequence,
        ]

    @rowproperty
    def video(self):
        sequence = (pipe_stim.GaborSequence & self.key).fetch1()
        fps = (pipe_gabor.Display & sequence).fetch1("fps")

        movs = pipe_gabor.Sequence.Gabor * pipe_gabor.Gabor & sequence
        movs = movs.fetch("movie", order_by="sequence_id ASC")
        assert len(movs) == sequence["sequence_length"]

        return video.Video.fromarray(np.concatenate(movs), period=1 / fps)


@keys
class DotSequence(Video):
    """Dot Sequence Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.DotSequence,
        ]

    @rowproperty
    def video(self):
        sequence = (pipe_stim.DotSequence & self.key).fetch1()
        fps = (pipe_dot.Display & sequence).fetch1("fps")

        imgs = pipe_dot.Dot * pipe_dot.Sequence.Dot * pipe_dot.Display & sequence
        imgs = imgs.fetch("image", order_by="dot_id ASC")
        assert len(imgs) == sequence["sequence_length"]

        n_frames, id_trace = (pipe_dot.Trace * pipe_dot.Display & sequence).fetch1("n_frames", "id_trace")
        frames = np.stack(imgs[id_trace])
        assert len(frames) == n_frames

        return video.Video.fromarray(frames, period=1 / fps)


@keys
class RdkSequence(Video):
    """Rdk Sequence Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.RdkSequence,
        ]

    @rowproperty
    def video(self):
        sequence = (pipe_stim.RdkSequence & self.key).fetch1()
        fps = (pipe_rdk.Display & sequence).fetch1("fps")

        movs = []
        for i in range(sequence["sequence_length"]):
            _key = dict(sequence, sequence_id=i)
            if pipe_rdk.Sequence.Rotation & _key:
                movie = (pipe_rdk.RotationRdk * pipe_rdk.Sequence.Rotation & _key).fetch1("movie")
            elif pipe_rdk.Sequence.Radial & _key:
                movie = (pipe_rdk.RadialRdk * pipe_rdk.Sequence.Radial & _key).fetch1("movie")
            elif pipe_rdk.Sequence.Translation & _key:
                movie = (pipe_rdk.TranslationRdk * pipe_rdk.Sequence.Translation & _key).fetch1("movie")
            else:
                raise Exception
            movs += [movie]

        return video.Video.fromarray(np.concatenate(movs), period=1 / fps)


@keys
class Frame(Video):
    """Frame Video"""

    @property
    def key_list(self):
        return [
            pipe_stim.Frame,
        ]

    @rowproperty
    def video(self):
        tup = pipe_stim.StaticImage.Image * pipe_stim.Frame & self.key
        image, pre_blank, duration = tup.fetch1("image", "pre_blank_period", "presentation_time")
        image = video.Frame.fromarray(image)

        if image.mode == "L":
            blank = np.full([image.height, image.width], 128, dtype=np.uint8)
            blank = video.Frame.fromarray(blank)
        else:
            raise NotImplementedError(f"Frame mode {mode} not implemented")

        if pre_blank > 0:
            return video.Video([blank, image, blank], times=[0, pre_blank, pre_blank + duration])
        else:
            return video.Video([image, blank], [0, duration])


# -- Transformed Video --


@keys
class ResizedVideo:
    """Resize video"""

    @property
    def key_list(self):
        return [
            stimulus.Video,
            utility.Resize,
            utility.Resolution,
        ]

    @rowproperty
    def video(self):
        from foundation.utility.resize import Resize, Resolution
        from foundation.stimulus.video import Video

        # load video
        video = (Video & self.key).link.compute.video

        # target size
        height, width = (Resolution & self.key).fetch1("height", "width")

        # resize video
        return (Resize & self.key).link.resize(video, height, width)
