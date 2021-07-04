import sys

#sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import numpy as np
import threading
from threading import Thread, Lock, Condition

# import cloudpickle
from queue import Queue
import csv
import time
from itertools import chain
from collections import defaultdict

# import multiprocessing
from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing
import pickle
import g2o
import json
from enum import Enum
import numpy as np
import time
from itertools import chain
from collections import defaultdict
from tracker import find_all_corners_from_bboxs

from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement, MapPoint
from motion import MotionModel
from loopclosing import LoopClosing
from camera_frame_test import *
from world_frame_gps import *
from world_coord_place_semantic_database import get_world_coords_from_place_semantic_database, place_semantic_db
from detect_place import get_place
from corresponding_corner import corresp_corner
import json
import traceback
import cv2
import ast
from config import corner_gt, characteristics

from correspondence_matcher import correspondence_matcher

# SEMANTIC_DATABASE_FILE_AVERAGE = "../set1/world_frame_set_1_average.txt"
# SEMANTIC_DATABASE_FILE = "../set1/world_frame_set_1.txt"

char_obj_file = f'{corner_gt}'
with open(char_obj_file, "rb") as f:
    char_obj_list = json.load(f)


class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        #print(len(measurements))
        assert len(measurements) >= self.min_measurements, "Not enough points"

        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)


class SPTAM(object):
    def __init__(self, params):
        self.params = params
        self._lock = threading.Lock()
        # infile1=open('trackerfile','rb')
        # self.tracker=pickle.load(infile1,encoding='bytes')
        self.tracker = Tracking(params)
        # infile2=open('motionfile','rb')
        # self.motion_model=pickle.load(infile2,encoding='bytes')
        self.motion_model = MotionModel()
        # infile3 = open('RY','rb')
        self.graph = CovisibilityGraph()
        # self.graph=dill.load(infile3)
        # self.graph = CovisibilityGraph()
        # infile4 = open('mappingfile','rb')
        # self.mapping=dill.load(infile4)

        self.mapping = MappingThread(self.graph, params)
        # infile5 = open('loopclosurefile','rb')
        # self.loop_closing=pickle.load(infile5,encoding='bytes')

        self.loop_closing = LoopClosing(self, params)
        self.loop_correction = None

        self.reference = None  # reference keyframe
        self.preceding = None  # last keyframe
        self.current = None  # current frame
        self.status = defaultdict(bool)
        # self.graph=threading.Lock()

    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def dumper(self):
        # with self._lock:
        # print("track",self.graph)
        dill.dump(self.tracker, open("trackerfile", "wb"))
        dill.dump(self.motion_model, open("motionfile", "wb"))
        # print(self.graph)
        dill.dump(self.graph, open("RY", "wb"))
        # dill.dump(self.graph.kfs, open('graphfile_kfs','wb'))
        # dill.dump(self.graph.pts, open('graphfile_pts','wb'))
        # dill.dump(self.graph.kfs_set, open('graphfile_set','wb'))
        # dill.dump(self.graph.meas_lookup, open('graphfile_meas_lookup','wb'))

        dill.dump(self.mapping, open("mappingfile", "wb"))
        dill.dump(self.loop_closing, open("loopclosurefile", "wb"))

    def initialize(
        self, frame, r_image, l_image, bboxes_left, bboxes_right, left_paths, right_paths,distance_threshhold
    ):
        mappoints, measurements = frame.triangulate()
        print(len(mappoints))
        assert len(mappoints) >= self.params.init_min_points, "Not enough points to initialize map."

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        #################################################################### CHANGES #######################################################################

        mappointsSemanticDatabase = []

        place = get_place(left_paths)
        char_image_name = characteristics[place]
        bboxes_char = char_obj_list[char_image_name]
        measurementsSemantics = []
        mappointsSemantics = []
        if bboxes_left is not None and len(bboxes_char) > 1 and len(bboxes_left) > 1:
            #print(bboxes_left)
            corresponding_bboxes = correspondence_matcher(bboxes_char, bboxes_left)
            #print(corresponding_bboxes)
            oi = 0
            bboxes_right = find_all_corners_from_bboxs(bboxes_right)
            corresponding_bboxes, bboxes_right = corresp_corner(corresponding_bboxes, bboxes_right)
            for i,char_bbox in corresponding_bboxes:
                oi = oi + 1
                # print("BBOX: ", i)
                
                var = compute_coord_robot(left_paths, right_paths, i, bboxes_right[oi-1])  # are we using arbitrary number of corners
                if var != None:
                    # print("OK")
                    (kps_left, kps_right, descs_left, descs_right, image_coord_left, image_coord_right) = var
                    if not (kps_left is None) and not (frame is None):
                        name_obj = i[0]
                        char_name_obj = char_bbox[0]
                        # fpos = coord_world(oi, matrx, frame.pose, name_obj, left_paths, right_paths, image_coord,)
                        # name_obj = i[0]
                        
                        # fpos = coord_world(oi, matrx, frame.pose, name_obj, left_paths, right_paths, image_coord)
                        for j in range(min(len(kps_left), len(kps_right))):  # which corner?
                            # world_coord = np.array(fpos["Pos" + str(j + 1)])[:3, -1]
                            # robot_coord = np.array(matrx[j]).reshape(3, 1)
                            corner = np.array(image_coord_left[j]).reshape(2,1)
                            world_coord = get_world_coords_from_place_semantic_database(corner, place, char_name_obj, "Pos" + str(j + 1))
                            #print(world_coord, frame.position)
                            if(world_coord is None):
                                continue
                            world_coord = world_coord.ravel()
                            temp = world_coord - frame.position
                            #print(world_coord.shape, frame.position.shape)
                            mp = MapPoint(world_coord, temp / np.linalg.norm(temp), descs_left[j])
                            mappointsSemanticDatabase.append((mp, kps_left[j], descs_left[j], kps_right[j], descs_right[j]))

            # if database_left:
            #     for obj in database_left:
            #         if obj:
            #             for i in obj:
            #                 if i[:3] == 'Pos' and len(i)==4:
            #                     matrx = np.array(obj[i])
            #                     temp = matrx[:3,-1] - keyframe.position
            #                     mp = MapPoint(matrx[:3,-1],temp/np.linalg.norm(temp),obj[i+'desc_l'])
            #                     l_tup = obj[i+'kps_l']
            #                     r_tup = obj[i+'kps_r']
            #                     kps_l = cv2.KeyPoint(x = l_tup[0][0],y = l_tup[0][1],_size = l_tup[1],_angle = l_tup[2],_response = l_tup[3],_octave = l_tup[4],_class_id = l_tup[5])
            #                     kps_r = cv2.KeyPoint(x = r_tup[0][0],y = r_tup[0][1],_size = r_tup[1],_angle = r_tup[2],_response = r_tup[3],_octave = r_tup[4],_class_id = r_tup[5])
            #                     mappointsSemanticDatabase.append((mp,kps_l,obj[i+'desc_l'],kps_r,obj[i+'desc_r']))

            measurementsSemantics = []
            mappointsSemantics = []
            for (originalMappoint, kp_l, desc_l, kp_r, desc_r,) in mappointsSemanticDatabase:
                kp_l.pt = (kp_l.pt[0], r_image.shape[0] - kp_l.pt[1])
                kp_r.pt = (kp_r.pt[0], r_image.shape[0] - kp_r.pt[1])
                measSemantic = Measurement(Measurement.Type.STEREO, Measurement.Source.SEMANTIC, [kp_l, kp_r], [desc_l, desc_r],)
                measSemantic.mappoint = originalMappoint
                # print("------------------------------------------------------------------------------")
                # print("SEMANTIC MAPPOIINTS: ", originalMappoint.position)
                # print("------------------------------------------------------------------------------")
                measSemantic.keyframe = keyframe
                measurementsSemantics.append(measSemantic)
                mappointsSemantics.append(originalMappoint)

            self.mapping.add_measurements(keyframe, mappointsSemantics, measurementsSemantics)

        ######################################################################################################################################################

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status["initialized"] = True

        self.motion_model.update_pose(frame.timestamp, frame.position, frame.orientation)

    def track(
        self, frame, r_image, l_image, bboxes_left, bboxes_right, left_paths, right_paths, distance_threshhold
    ):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        # print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(frame.orientation, frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        #print("MEASUREMENTS KIIIIIIIIIII LENGTH", len(local_mappoints))
        if len(local_mappoints) <= 0:
            return False
        # print(
        #     g2o.SBACam(frame.pose.orientation(), frame.pose.position()).to_homogeneous_matrix()[0][3],
        #     g2o.SBACam(frame.pose.orientation(), frame.pose.position()).to_homogeneous_matrix()[1][3],
        #     g2o.SBACam(frame.pose.orientation(), frame.pose.position()).to_homogeneous_matrix()[2][3],
        # )
        measurements = frame.match_mappoints(local_mappoints, Measurement.Source.TRACKING)

        #################################################################### CHANGES #######################################################################

        # mappointsSemanticDatabase = []
        # if bboxes_left is not None:
        #     for i in bboxes_left:
        #         var = compute_coord_robot(left_paths,right_paths,i)
        #         if var!=None:
        #             print("OK")
        #             matrx,kps_left,kps_right,descs_left,descs_right = var
        #             if(matrx is not None and kps_left is not None and frame is not None):
        #                 fpos = coord_world(matrx,frame.pose)
        #                 for j in range(min(len(kps_left), len(kps_right))):
        #                     temp = fpos["Pos"+str(j+1)][:3,-1]-frame.position
        #                     mp = MapPoint(fpos["Pos"+str(j+1)][:3,-1],temp/np.linalg.norm(temp),descs_left[j])
        #                     # print(kps_right)
        #                     mappointsSemanticDatabase.append((mp,kps_left[j],descs_left[j],kps_right[j],descs_right[j]))
        mappointsSemanticDatabase = []

        place = get_place(left_paths)
        char_image_name = characteristics[place]
        bboxes_char = char_obj_list[char_image_name]
        measurementsSemantics = []
        mappointsSemantics = []
        if bboxes_left is not None and len(bboxes_char) > 1 and len(bboxes_left) > 1:
        	
            place = get_place(left_paths)
            char_image_name = characteristics[place]
            bboxes_char = char_obj_list[char_image_name]
            #print(bboxes_left)
            corresponding_bboxes = correspondence_matcher(bboxes_char, bboxes_left)
            oi = 0
            #print(corresponding_bboxes)
            #bboxes_left = find_all_corners_from_bboxs(bboxes_left[left_images[i]])
            bboxes_right = find_all_corners_from_bboxs(bboxes_right)
            corresponding_bboxes, bboxes_right = corresp_corner(corresponding_bboxes, bboxes_right)
            for i,char_bbox in corresponding_bboxes:
                oi = oi + 1
                # print("BBOX: ", i)
                
                var = compute_coord_robot(left_paths, right_paths, i, bboxes_right[oi-1])  # are we using arbitrary number of corners
                if var != None:
                    # print("OK")
                    (kps_left, kps_right, descs_left, descs_right, image_coord_left, image_coord_right) = var
                    if not (kps_left is None) and not (frame is None):
                        name_obj = i[0]
                        char_name_obj = char_bbox[0]
                        # fpos = coord_world(oi, matrx, frame.pose, name_obj, left_paths, right_paths, image_coord,)
                        # name_obj = i[0]
                        
                        # fpos = coord_world(oi, matrx, frame.pose, name_obj, left_paths, right_paths, image_coord)
                        for j in range(min(len(kps_left), len(kps_right))):  # which corner?
                            # world_coord = np.array(fpos["Pos" + str(j + 1)])[:3, -1]
                            # robot_coord = np.array(matrx[j]).reshape(3, 1)
                            corner = np.array(image_coord_left[j]).reshape(2,1)
                            world_coord = get_world_coords_from_place_semantic_database(corner, place, char_name_obj, "Pos" + str(j + 1))
                            #print(world_coord, frame.position)
                            if(world_coord is None):
                                continue
                            world_coord = world_coord.ravel()
                            temp = world_coord - frame.position
                            #print(world_coord.shape, frame.position.shape)
                            mp = MapPoint(world_coord, temp / np.linalg.norm(temp), descs_left[j])
                            mappointsSemanticDatabase.append((mp, kps_left[j], descs_left[j], kps_right[j], descs_right[j]))

            # if database_left:
            #     for obj in database_left:
            #         if obj:
            #             for i in obj:
            #                 if i[:3] == 'Pos' and len(i)==4:
            #                     matrx = np.array(obj[i])
            #                     temp = matrx[:3,-1] - frame.position
            #                     mp = MapPoint(matrx[:3,-1],temp/np.linalg.norm(temp),obj[i+'desc_l'])
            #                     l_tup = obj[i+'kps_l']
            #                     r_tup = obj[i+'kps_r']
            #                     kps_l = cv2.KeyPoint(x = l_tup[0][0],y = l_tup[0][1],_size = l_tup[1],_angle = l_tup[2],_response = l_tup[3],_octave = l_tup[4],_class_id = l_tup[5])
            #                     kps_r = cv2.KeyPoint(x = r_tup[0][0],y = r_tup[0][1],_size = r_tup[1],_angle = r_tup[2],_response = r_tup[3],_octave = r_tup[4],_class_id = r_tup[5])
            #                     mappointsSemanticDatabase.append((mp,kps_l,obj[i+'desc_l'],kps_r,obj[i+'desc_r']))

            measurementsSemantics = []
            mappointsSemantics = []
            for (originalMappoint, kp_l, desc_l, kp_r, desc_r,) in mappointsSemanticDatabase:
                kp_l.pt = (kp_l.pt[0], r_image.shape[0] - kp_l.pt[1])
                kp_r.pt = (kp_r.pt[0], r_image.shape[0] - kp_r.pt[1])
                measSemantic = Measurement(Measurement.Type.STEREO, Measurement.Source.SEMANTIC, [kp_l, kp_r], [desc_l, desc_r],)
                measSemantic.mappoint = originalMappoint
                # print("------------------------------------------------------------------------------")
                # print("SEMANTIC MAPPOIINTS: ", originalMappoint.position)
                # print("------------------------------------------------------------------------------")
                measSemantic.keyframe = frame
                measurementsSemantics.append(measSemantic)
                mappointsSemantics.append(originalMappoint)

        tracked_map = set()

        # for m in measurementsSemantics:
        #     mappoint = m.mappoint
        #     mappoint.update_descriptor(m.get_descriptor())
        #     mappoint.increase_measurement_count()
        #     tracked_map.add(mappoint)

        ##########################################################################################################################################################

        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)
        # print("######################", len(tracked_map))

        try:
            self.reference = self.graph.get_reference_frame(tracked_map)
            #print("abs")
            f = open("./output_files/final_positions.txt", "a+")
            # hypothesis A and B   
            print(len(measurementsSemantics), len(measurements)) 
            if len(measurements)>1: 
                pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements + measurementsSemantics)    
            else:   
                pose = self.tracker.refine_pose(frame.pose, frame.cam, measurementsSemantics)
            frame.update_pose(pose)
            self.motion_model.update_pose(frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True

            data = {}
            print(frame.idx)
            data["ID"] = frame.idx
            data["Time"] = frame.timestamp
            data["Position"] = np.array(pose.position()).tolist()
            data["Matrix"] = np.array(pose.to_homogeneous_matrix()).tolist()
            json_data = json.dumps(data)
            f.write(json_data + "\n")
            # print(pose.to_homogeneous_matrix())

            # print(np.array(pose.position()).tolist)
        except Exception as e:
            print(e)
            tracking_is_ok = False
            # print('tracking failed!!!')
            # print(traceback.format_exc())

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):

            # print("########################################### keyframe addded #################################################")
            # print('new keyframe', frame.idx)

            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)
            # self.mapping.add_keyframe(keyframe, measurementsSemantics)
            ######################################################################## CHANGES ###########################################################################
            # measurements.extend(measurementsSemantics)
            # for m in temp_list:
            # 	print(m.source == Measurement.Source.SEMANTIC)
            # self.mapping.add_keyframe(keyframe, temp_list)
            self.mapping.add_keyframe(keyframe, measurements)
            self.mapping.add_measurements(keyframe, mappointsSemantics, measurementsSemantics)

            ############################################################################################################################################################

            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe

        self.set_tracking(False)
        return True

    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2([self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        # print('filter points:', len(local_mappoints), can_view.sum(),
        #    len(self.preceding.mappoints()),
        #    len(self.reference.mappoints()))

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered

    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        # print("NMATCHES: ", n_matches)
        n_matches_ref = len(self.reference.measurements())
        # print("NMATCHES REFERENCE: ", n_matches_ref)
        # print('keyframe check:', n_matches, '   ', n_matches_ref)

        return ((n_matches / n_matches_ref) < self.params.min_tracked_points_ratio) or n_matches < 20

    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        return self.status["initialized"]

    def pause(self):
        self.status["paused"] = True

    def unpause(self):
        self.status["paused"] = False

    def is_paused(self):
        return self.status["paused"]

    def is_tracking(self):
        return self.status["tracking"]

    def set_tracking(self, status):
        self.status["tracking"] = status

    def stop_adding_keyframes(self):
        self.status["adding_keyframes_stopped"] = True

    def resume_adding_keyframes(self):
        self.status["adding_keyframes_stopped"] = False

    def adding_keyframes_stopped(self):
        return self.status["adding_keyframes_stopped"]


if __name__ == "__main__":
    import cv2
    import g2o

    import os
    import sys
    import argparse

    from threading import Thread

    from components import Camera
    from components import StereoFrame
    from feature import ImageFeature
    from params import ParamsKITTI, ParamsEuroc
    from dataset import KITTIOdometry, EuRoCDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viz", action="store_true", help="do not visualize")
    parser.add_argument("--dataset", type=str, help="dataset (KITTI/EuRoC)", default="KITTI")
    parser.add_argument(
        "--path", type=str, help="dataset path", default="path/to/your/KITTI_odometry/sequences/00",
    )
    args = parser.parse_args()

    if args.dataset.lower() == "kitti":
        params = ParamsKITTI()
        dataset = KITTIOdometry(args.path)
    elif args.dataset.lower() == "euroc":
        params = ParamsEuroc()
        dataset = EuRoCDataset(args.path)

    sptam = SPTAM(params)

    visualize = not args.no_viz
    if visualize:
        from viewer import MapViewer

        viewer = MapViewer(sptam, params)

    cam = Camera(
        dataset.cam.fx,
        dataset.cam.fy,
        dataset.cam.cx,
        dataset.cam.cy,
        dataset.cam.width,
        dataset.cam.height,
        params.frustum_near,
        params.frustum_far,
        dataset.cam.baseline,
    )

    durations = []

    if os.path.exists("Positions_v2.txt"):
        os.remove("Positions_v2.txt")

    with open("./final_corners.json") as f:
        bboxes_json = json.load(f)
        f.close()

    with open("./final_corners_right.json") as f:
        bboxes_json_right = json.load(f)
        f.close()

    # f = open("cluters_world_cordis.txt")
    # f = f.read()
    # SemanticDatabase = ast.literal_eval(f)
    semantic_database = []

    # with open(SEMANTIC_DATABASE_FILE, "r") as f:
    #     for line in f:
    #         semantic_database.append(json.loads(line))
    #     f.close()

    # with open(SEMANTIC_DATABASE_FILE_AVERAGE, "r") as f:
    #     average_semantic_database = json.loads(f.read())

    for i in range(len(dataset))[:10000]:
        # print(dataset.left[i])
        featurel = ImageFeature(dataset.left[i], params)
        featurer = ImageFeature(dataset.right[i], params)
        timestamp = dataset.timestamps[i]

        time_start = time.time()

        t = Thread(target=featurer.extract)
        t.start()
        featurel.extract()
        t.join()

        frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

        ######################################################################## CHANGES ###########################################################################

        left_images = ["left" + name[-10:] for name in dataset.left_images]
        right_images = ["right" + name[-10:] for name in dataset.right_images]

        # print("IMAGE NAME: ", left_images[i])
        # print("BBOXES Left: ", bboxes_json[left_images[i]])
        # print("BBOXES Right: ", bboxes_json[right_images[i]])

        #bboxes_left = find_all_corners_from_bboxs(left_images[i], bboxes_json[left_images[i]])
        #bboxes_right = find_all_corners_from_bboxs(right_images[i], bboxes_json_right[right_images[i]])




        # print("mapoints" "measurements", frame.triangulate())
        distance_threshhold = 10    
        distance_threshhold_min = 5
        if not sptam.is_initialized():
            sptam.initialize(
                frame,
                dataset.left[i],
                dataset.right[i],
                bboxes_json[left_images[i]],
                bboxes_json_right[right_images[i]],
                left_images[i],
                right_images[i],
                distance_threshhold
            )
            # sptam.dumper()
        else:
            if not sptam.track(
                frame,
                dataset.left[i],
                dataset.right[i],
                bboxes_json[left_images[i]],
                bboxes_json_right[right_images[i]],
                left_images[i],
                right_images[i],
                distance_threshhold
            ):
                sptam.initialize(
                    frame,
                    dataset.left[i],
                    dataset.right[i],
                    bboxes_json[left_images[i]],
                	bboxes_json_right[right_images[i]],
                    left_images[i],
                    right_images[i],
                    distance_threshhold
                )
                # sptam.dumper()

        ############################################################################################################################################################

        duration = time.time() - time_start
        durations.append(duration)
        # print('duration', duration)
        # print()
        # print()

        if visualize:
            viewer.update()

    print("num frames", len(durations))
    print("num keyframes", len(sptam.graph.keyframes()))
    print("average time", np.mean(durations))

    f.close()
    sptam.stop()
    if visualize:
        viewer.stop()
    sptam.dumper()

