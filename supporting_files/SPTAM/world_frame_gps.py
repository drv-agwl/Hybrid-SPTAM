import json
import numpy as np
import math
import g2o


cameraFrame = []
worldFrame = []
data = []
Xroad = []
Yroad = []
finalPos = {}
sync = []
theta = np.radians(0)
c, s = np.cos(theta), np.sin(theta)
rotation = [[1.0,0.0,0.0,0.0],
[0.0,c,-s,0.0],
[0.0,s,c,0.0]]


#gt = pd.read_csv(ground_truth, names = ["X", "Y", "Z","Time"])
def coord_world(oi,Mat, Position1, name, left, right, image_coord):
	#print(int(cameraFrame[i]['ID']))
	# del_x = Position1[0] - Position2[0]
	# del_y = Position1[1] - Position2[1]
	# angle = math.atan2(del_y,del_x)
	# sin, cos = np.sin(angle), np.cos(angle)
	# print(Position1)
	f = open("./finalPosition2.txt", "a+")
	rot = g2o.SBACam(Position1.orientation(), Position1.position()).to_homogeneous_matrix()
	# [[cos,-sin,0.0,worldFrame[index]["Position"][0]],[sin, cos,0.0,worldFrame[index]["Position"][1]],[0.0,0.0,1.0,worldFrame[index]["Position"][2]],[0.0,0.0,0.0,1.0]]
	mat = Mat

	finalPos["ID"] = left
	finalPos["Name"] = name
	finalPos["oi"] = oi
	for i in range(len(mat)):
		pos = [[1.0,0.0,0.0,mat[i][0]],[0.0,1.0,0.0,mat[i][1]],[0.0,0.0,1.0,mat[i][2]],[0.0,0.0,0.0,1.0]]
		finalPos['Pos'+str(i+1)] = np.dot(rot, pos).tolist()
		finalPos['Posxy' + str(i+1)] = image_coord[i]

	# pos1 = [[1.0,0.0,0.0,mat[0][0][3]],[0.0,1.0,0.0,mat[0][1][3]],[0.0,0.0,1.0,mat[0][2][3]],[0.0,0.0,0.0,1.0]]
	# pos2 = [[1.0,0.0,0.0,mat[1][0][3]],[0.0,1.0,0.0,mat[1][1][3]],[0.0,0.0,1.0,mat[1][2][3]],[0.0,0.0,0.0,1.0]]
	# pos3 = [[1.0,0.0,0.0,mat[2][0][3]],[0.0,1.0,0.0,mat[2][1][3]],[0.0,0.0,1.0,mat[2][2][3]],[0.0,0.0,0.0,1.0]]
	# pos4 = [[1.0,0.0,0.0,mat[3][0][3]],[0.0,1.0,0.0,mat[3][1][3]],[0.0,0.0,1.0,mat[3][2][3]],[0.0,0.0,0.0,1.0]]
	#pos = [[1.0,0.0,0.0,cameraFrame[i]['X']],[0.0,1.0,0.0,cameraFrame[i]['Y']],[0.0,0.0,1.0,cameraFrame[i]['Z']],[0.0,0.0,0.0,1.0]]
#rot = [[1.0,0.0,0.0,0.0],[0.0,cos,-sin,0.0],[0.0,sin,cos,0.0]]
	#position = [cameraFrame[i]['Z'], cameraFrame[i]['Y'], cameraFrame[i]['X'],1.0]
	#finalPosition =np.dot(rot, np.dot(np.array(worldFrame[int(cameraFrame[i]['ID'])]['Mat']),position))
	# finalPosition1 = np.dot(rot,pos1)
	# finalPosition2 = np.dot(rot,pos2)
	# finalPosition3 = np.dot(rot,pos3)
	# finalPosition4 = np.dot(rot,pos4)
	# # finalPos["ID"] = cameraFrame[i]['ID']
	# # finalPos["Name"] = cameraFrame[i]['Name']
	# finalPos["Pos1"] = finalPosition1
	# finalPos["Pos2"] = finalPosition2
	# finalPos["Pos3"] = finalPosition3
	# finalPos["Pos4"] = finalPosition4
	json_data = json.dumps(finalPos)
	f.write(json_data+str("\n"))
	#print("@@@@",finalPos)

	return finalPos
