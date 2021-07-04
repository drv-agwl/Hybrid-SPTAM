from threading import Lock

from collections import defaultdict, Counter
from itertools import chain
import pickle
class GraphKeyFrame(object):
    def __init__(self):
        self.id = None
        self.meas = dict()
        self.covisible = defaultdict(int)
        self._lock = Lock()
    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphKeyFrame) and
            self.id == rhs.id)
    def __lt__(self, rhs):
        return self.id < rhs.id   # predate
    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):
        with self._lock:
            return self.meas.keys()

    def mappoints(self):
        with self._lock:
            return self.meas.values()

    def add_measurement(self, m):
        with self._lock:
            self.meas[m] = m.mappoint

    def remove_measurement(self, m):
        with self._lock:
            try:
                del self.meas[m]
            except KeyError:
                pass

    def covisibility_keyframes(self):
        with self._lock:
            return self.covisible.copy()  # shallow copy

    def add_covisibility_keyframe(self, kf):
        with self._lock:
            self.covisible[kf] += 1
    def __getstate__(self):
        #print(33333)
        abc=self
        abc.meas=tuple(self.meas)
        abc.covisible=tuple(self.covisible);
        return abc.__dict__

    def __setstate__(self, dct):
        #print(33333)
        self.__dict__=dct
        self.meas=dict()
        self.covisible=dict()
        for s in dct['meas']:
            self.meas.add(s)
        #del self.meas
        for i in dct['covisible']:
        	self.covisible.add(i)
        #del self.covisible


class GraphMapPoint(object):
    def __init__(self):
        self.id = None
        self.meas = dict()
        self._lock = Lock()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMapPoint) and
            self.id == rhs.id)
    def __lt__(self, rhs):
        return self.id < rhs.id
    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):
        with self._lock:
            return self.meas.keys()

    def keyframes(self):
        with self._lock:
            return self.meas.values()

    def add_measurement(self, m):
        with self._lock:
            self.meas[m] = m.keyframe

    def remove_measurement(self, m):
        with self._lock:
            try:
                del self.meas[m]
            except KeyError:
                pass

    def __getstate__(self):
        abc=self
        abc.meas=tuple(self.meas)
        return abc.__dict__

    def __setstate__(self, dct):
        #print(22222)
        self.__dict__=dct
        self.meas=dict()
        for s in dct['meas']:
            self.meas.add(s)
        #del self.meas




class GraphMeasurement(object):
    def __init__(self):
        self.keyframe = None
        self.mappoint = None

    @property
    def id(self):
        return (self.keyframe.id, self.mappoint.id)

    def __hash__(self):
        return hash(self.id)


    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMeasurement) and
            self.id == rhs.id)

    def __getstate__(self):
        abc=self.__dict__
        return abc

    def __setstate__(self, dct):
        #print(11111)
        self.__dict__=dct





class CovisibilityGraph(object):
    def __init__(self, ):
        self._lock = Lock()
        #infile1=open('graphfile_kfs','rb')
        #self.kfs=pickle.load(infile1,encoding='bytes')
        self.kfs = []
        #infile2=open('graphfile_pts','rb')
        #self.pts=pickle.load(infile2,encoding='bytes')
        self.pts = set()

        self.kfs_set = set()
        #infile3=open('graphfile_meas_lookup','rb')
        #self.meas_lookup=pickle.load(infile3,encoding='bytes')
        self.meas_lookup = dict()


    def __getstate__(self):
        abc=self
        abc.kfs=self.kfs
        abc.kfs_set=tuple(self.kfs_set)
        abc.pts=self.pts
        abc.meas_lookup=tuple(self.meas_lookup)
        return abc.__dict__


        '''self.links= set()
        for link in state:
            link.base= self # set the missing attribute
            self.links.add(link) # now it can be hashed'''

    def __setstate__(self, dct):
        self._lock = Lock()
        self.__dict__=dct
        self.kfs = []
        self.pts = set()
        self.kfs_set = set()
        self.meas_lookup = dict()

        for i in dct['kfs_set']:
        	#print(i)
        	self.kfs_set.add(i)
        for s in dct['meas_lookup']:
        	#print(s)
        	self.meas_lookup.add(s)
        #del self.meas_lookup
        #del self.kfs_set


    def keyframes(self):
        with self._lock:
            return self.kfs.copy()

    def mappoints(self):
        with self._lock:
            return self.pts.copy()

    def add_keyframe(self, kf):
        with self._lock:
            self.kfs.append(kf)
            self.kfs_set.add(kf)

    def add_mappoint(self, pt):
        with self._lock:
            self.pts.add(pt)

    def remove_mappoint(self, pt):
        with self._lock:
            try:
                for m in pt.measurements():
                    m.keyframe.remove_measurement(m)
                    del self.meas_lookup[m.id]
                self.pts.remove(pt)
            except:
                pass

    def add_measurement(self, kf, pt, meas):
        with self._lock:
            if kf not in self.kfs_set or pt not in self.pts:
                return

            for m in pt.measurements():
                if m.keyframe == kf:
                    continue
                kf.add_covisibility_keyframe(m.keyframe)
                m.keyframe.add_covisibility_keyframe(kf)

            meas.keyframe = kf
            meas.mappoint = pt
            kf.add_measurement(meas)
            pt.add_measurement(meas)

            self.meas_lookup[meas.id] = meas

    def remove_measurement(self, m):
        m.keyframe.remove_measurement(m)
        m.mappoint.remove_measurement(m)
        with self._lock:
            try:
                del self.meas_lookup[m.id]
            except:
                pass

    def has_measurement(self, *args):
        with self._lock:
            if len(args) == 1:                                 # measurement
                return args[0].id in self.meas_lookup
            elif len(args) == 2:                               # keyframe, mappoint
                id = (args[0].id, args[1].id)
                return id in self.meas_lookup
            else:
                raise TypeError

    def get_reference_frame(self, seedpoints):
        assert len(seedpoints) > 0
        visible = [pt.keyframes() for pt in seedpoints]
        visible = Counter(chain(*visible))
        return visible.most_common(1)[0][0]

    def get_local_map(self, seedpoints, window_size=15):
        reference = self.get_reference_frame(seedpoints)
        covisible = chain(
            reference.covisibility_keyframes().items(), [(reference, float('inf'))])
        covisible = sorted(covisible, key=lambda _:_[1], reverse=True)

        local_map = [seedpoints]
        local_keyframes = []
        for kf, n in covisible[:window_size]:
            if n < 1:
                continue
            local_map.append(kf.mappoints())
            local_keyframes.append(kf)
        local_map = list(set(chain(*local_map)))

        return local_map, local_keyframes

    def get_local_map_v2(self, seedframes, window_size=12, loop_window_size=8):
        covisible = []
        for kf in set(seedframes):
            covisible.append(Counter(kf.covisibility_keyframes()))
        covisible = sum(covisible, Counter())
        for kf in set(seedframes):
            covisible[kf] = float('inf')
        local = sorted(
            covisible.items(), key=lambda _:_[1], reverse=True)

        id = max([_.id for _ in covisible])
        loop_frames = [_ for _ in local if _[0].id < id-10]
        #print("Loop Frames : ", loop_frames, "\nLocal : " , local)
        local = local[:window_size]
        loop_local = []
        if len(loop_frames) > 0:
            loop_covisible = sorted(
                loop_frames[0][0].covisibility_keyframes().items(),
                key=lambda _:_[1], reverse=True)

            for kf, n in loop_covisible:
                if kf not in set([_[0] for _ in local]):
                    loop_local.append((kf, n))
                    if len(loop_local) >= loop_window_size:
                        break

        local = chain(local, loop_local)

        local_map = []
        local_keyframes = []
        for kf, n in local:
            if n < 1:
                continue
            local_map.append(kf.mappoints())
            local_keyframes.append(kf)
        local_map = list(set(chain(*local_map)))

        return local_map, local_keyframes
