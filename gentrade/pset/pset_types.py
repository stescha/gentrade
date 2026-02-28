import numpy as np
import random



class NumericSeries:
    pass

class PriceSeries(NumericSeries):
    pass

class Open(PriceSeries):
    pass

class High(PriceSeries):
    pass

class Low(PriceSeries):
    pass

class Close(PriceSeries):
    pass

class Volume(NumericSeries):
    pass



class BooleanSeries:
    pass


class OHLCVFrame:
    pass


class Timeperiod:

    @staticmethod
    def sample():
        ts = [3,5,6, 7, 8, 9, 10, 12, 16, 20, 24, 30, 36, 42, 48, 52, 56, 60, 66, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 520, 528, 536, 544, 552, 560, 568, 576, 584, 592, 600, 608, 616, 624]
        return random.choice(ts)
#        return random.randint(2, 1000)



class NBDev:

    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.5, 5.1, 0.25).round(2)))


class MAType:


    @classmethod
    def sample(cls):
        # ma_type == 7 can cause unstable results for BBANDS calculations
        return random.choice([0, 1, 2, 3, 4, 5, 6, 8])

class FastLimit:
    
    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.1, 0.9, 0.05)))

class SlowLimit:

    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.01, 0.1, 0.01)))

class Acceleration:
    
    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.001, 0.08, 0.001)))
    
class Maximum:
    
    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.1, 0.5, 0.1)))

class  VFactor:

    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.1, 1, 0.1)))


class ZeroHalf:

    s = np.concatenate([np.arange(0.001, 0.051, 0.001), np.arange(0.06, 0.51, 0.01)]).round(3)

    @classmethod
    def sample(cls):
        return float(random.choice(cls.s))


class ZeroOneFine:

    param_names = ['vfactor']
    s = np.arange(0.01, 1, 0.01).round(3)

    @classmethod
    def sample(cls):
        return float(random.choice(cls.s))


class ZeroOneIncl:

    @classmethod
    def sample(cls):
        # return random.choice(np.arange(0.0, 1.05, 0.05))
        return float(random.choice(np.arange(0.05, 1.05, 0.05)))


class ZeroOneExcl:

    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0.05, 1.0, 0.05)))


class ZeroHundred:

    @classmethod
    def sample(cls):
        return float(random.choice(np.arange(0, 105, 5)))
        # return float(random.choice(np.arange(-105, 105, 5)))


class Threshold:
    """Ephemeral constant for zigzag threshold (0.01–0.10)."""

    @staticmethod
    def sample() -> float:
        return float(random.choice(np.arange(0.01, 0.1001, 0.005).round(3)))


class Label:
    """Ephemeral constant for zigzag label (-1 or 1)."""

    @staticmethod
    def sample() -> int:
        return random.choice([-1, 1])


