import operator
# from corresponding_corner import corresp_corner


def find_dist(bbox1, bbox2):
    Xcenter1 = (bbox1[1] + bbox1[3]) / 2
    Ycenter1 = (bbox1[2] + bbox1[4]) / 2

    Xcenter2 = (bbox2[1] + bbox2[3]) / 2
    Ycenter2 = (bbox2[2] + bbox2[4]) / 2

    return (Xcenter2 - Xcenter1) ** 2 + (Ycenter2 - Ycenter1) ** 2

def change_format(bbox_xmin):
    bbox_corner = []

    name = bbox_xmin[0]
    x1, y1 = bbox_xmin[1],bbox_xmin[2]
    x2, y2 = bbox_xmin[3],bbox_xmin[2]
    x3, y3 = bbox_xmin[1],bbox_xmin[4]
    x4, y4 = bbox_xmin[3],bbox_xmin[4]

    bbox_corner.extend([name, [x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    return bbox_corner

def correspondence_matcher(char_bbox_list, bbox_list):
    char_objs = len(char_bbox_list)
    char_mat = {}
    for i in range(len(char_bbox_list)):
        for j in range(i + 1, len(char_bbox_list)):
            dist = find_dist(char_bbox_list[i], char_bbox_list[j])
            char_mat[dist] = [i, j]
    max_dist = max(char_mat.keys(), key=(lambda k: k))
    char_mat_normalised = {}
    for key, val in char_mat.items():
        char_mat_normalised[key / max_dist] = val

    mat = {}
    for i in range(len(bbox_list)):
        for j in range(i + 1, len(bbox_list)):
            dist = find_dist(bbox_list[i], bbox_list[j])
            mat[dist] = [i, j]
    max_dist = max(mat.keys(), key=(lambda k: k))
    mat_normalised = {}
    for key, val in mat.items():
        mat_normalised[key / max_dist] = val

    assigned_pairs = {}  # objects --> char_objects
    for dist, obj_pair in mat_normalised.items():
        sorted_char_mat = sorted(char_mat_normalised.items(), key=lambda item: abs(item[0] - dist))
        query_img_objs = sorted([bbox_list[obj_idx][0] for obj_idx in obj_pair])

        for char_dist, char_obj_pair in sorted_char_mat:
            char_img_objs = sorted([char_bbox_list[obj_idx][0].split('_')[0] for obj_idx in char_obj_pair])
            if char_img_objs != query_img_objs:
                continue

            if bbox_list[obj_pair[0]][1] <= bbox_list[obj_pair[1]][1]:
                left_query_img_obj_idx = obj_pair[0]
                right_query_img_obj_idx = obj_pair[1]
            else:
                left_query_img_obj_idx = obj_pair[1]
                right_query_img_obj_idx = obj_pair[0]

            if char_bbox_list[char_obj_pair[0]][1] <= char_bbox_list[char_obj_pair[1]][1]:
                left_char_img_obj_idx = char_obj_pair[0]
                right_char_img_obj_idx = char_obj_pair[1]
            else:
                left_char_img_obj_idx = char_obj_pair[1]
                right_char_img_obj_idx = char_obj_pair[0]

            assigned_pairs[left_query_img_obj_idx] = left_char_img_obj_idx
            assigned_pairs[right_query_img_obj_idx] = right_char_img_obj_idx
            break

    correspondence_mat = []
    for obj_idx, char_obj_idx in assigned_pairs.items():
        #print(bbox_list[obj_idx],char_bbox_list[char_obj_idx])
        correspondence_mat.append([change_format(bbox_list[obj_idx]), change_format(char_bbox_list[char_obj_idx])])

    return correspondence_mat
